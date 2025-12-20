# train.py
"""
Training script for the Multi-Task Learning model.
This is the main entry point - run this file to train the model.
Includes masked loss calculation to handle NaN targets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import our custom modules
import importlib.util
from pathlib import Path

# Import numbered modules (Python can't import modules starting with numbers directly)
_package_dir = Path(__file__).parent

# Import config
_config_spec = importlib.util.spec_from_file_location("config", _package_dir / "01_config.py")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

# Import dataset
_dataset_spec = importlib.util.spec_from_file_location("dataset", _package_dir / "02_dataset.py")
dataset = importlib.util.module_from_spec(_dataset_spec)
_dataset_spec.loader.exec_module(dataset)

# Import model
_model_spec = importlib.util.spec_from_file_location("model", _package_dir / "03_model.py")
model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(model)

# Access classes from loaded modules
NHANESMultiTaskDataset = dataset.NHANESMultiTaskDataset
SharedBottomMTL = model.SharedBottomMTL


def masked_bce_loss(pred, target, mask, device):
    """
    Binary Cross Entropy loss that only considers valid (non-NaN) targets.
    
    Args:
        pred: Model predictions [batch, n]
        target: Ground truth labels [batch, n]
        mask: Valid target mask (1=valid, 0=was NaN) [batch, n]
        device: torch device (cuda or cpu)
    
    Returns:
        Mean loss over valid targets only
    """
    # Apply sigmoid and compute BCE manually to allow masking
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(pred, target)
    
    # Apply mask: zero out losses for NaN targets
    masked_loss = loss * mask
    
    # Average over valid targets only (avoid division by zero)
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=device)


def masked_mse_loss(pred, target, mask, device):
    """
    Mean Squared Error loss that only considers valid (non-NaN) targets.
    
    Args:
        pred: Model predictions [batch, 1]
        target: Ground truth values [batch, 1]
        mask: Valid target mask (1=valid, 0=was NaN) [batch, 1]
        device: torch device (cuda or cpu)
    
    Returns:
        Mean loss over valid targets only
    """
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    
    # Apply mask
    masked_loss = loss * mask
    
    # Average over valid targets only
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=device)


def train():
    """Main training function."""
    
    # ============================================================
    # 1. PREPARE DATA
    # ============================================================
    print("=" * 60)
    print("Loading Data...")
    print("=" * 60)
    
    train_dataset = NHANESMultiTaskDataset(
        db_path=config.DB_PATH,
        table_name='training_set',
        cont_cols=config.CONT_COLS,
        target_mapping=config.TARGET_MAPPING
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ============================================================
    # 3. INITIALIZE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("Initializing Model...")
    print("=" * 60)
    
    num_continuous = len(config.CONT_COLS)
    print(f"Number of input features: {num_continuous}")
    
    model = SharedBottomMTL(
        num_continuous=num_continuous,
        hidden_dim=config.HIDDEN_DIM
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============================================================
    # 4. SETUP OPTIMIZATION
    # ============================================================
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # ============================================================
    # 5. TRAINING LOOP
    # ============================================================
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Track individual task losses
        loss_cardio_sum = 0.0
        loss_metabolic_sum = 0.0
        loss_kidney_sum = 0.0
        loss_liver_sum = 0.0
        
        for batch in train_loader:
            # Unpack batch (now includes masks)
            x_cont, y_cardio, y_metabolic, y_kidney, y_liver, \
            mask_cardio, mask_metabolic, mask_kidney, mask_liver = batch
            
            # Move data to device
            x_cont = x_cont.to(device)
            y_cardio = y_cardio.to(device)
            y_metabolic = y_metabolic.to(device)
            y_kidney = y_kidney.to(device)
            y_liver = y_liver.to(device)
            mask_cardio = mask_cardio.to(device)
            mask_metabolic = mask_metabolic.to(device)
            mask_kidney = mask_kidney.to(device)
            mask_liver = mask_liver.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            p_cardio, p_metabolic, p_kidney, p_liver = model(x_cont)
            
            # Calculate masked losses (only valid targets contribute)
            loss_c = masked_bce_loss(p_cardio, y_cardio, mask_cardio, device)
            loss_m = masked_bce_loss(p_metabolic, y_metabolic, mask_metabolic, device)
            loss_k = masked_mse_loss(p_kidney, y_kidney, mask_kidney, device)
            loss_l = masked_mse_loss(p_liver, y_liver, mask_liver, device)
            
            # Combined loss (equal weighting)
            loss = loss_c + loss_m + loss_k + loss_l
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            loss_cardio_sum += loss_c.item()
            loss_metabolic_sum += loss_m.item()
            loss_kidney_sum += loss_k.item()
            loss_liver_sum += loss_l.item()
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_cardio = loss_cardio_sum / num_batches
        avg_metabolic = loss_metabolic_sum / num_batches
        avg_kidney = loss_kidney_sum / num_batches
        avg_liver = loss_liver_sum / num_batches
        
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Cardio: {avg_cardio:.4f} | "
              f"Metabolic: {avg_metabolic:.4f} | "
              f"Kidney: {avg_kidney:.4f} | "
              f"Liver: {avg_liver:.4f}")
    
    # ============================================================
    # 6. SAVE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Save the trained model
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
