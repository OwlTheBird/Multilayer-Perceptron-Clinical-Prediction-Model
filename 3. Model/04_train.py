# train.py
"""
Training script for the Multi-Task Learning model.
This is the main entry point - run this file to train the model.

Updated to include:
- Weighted BCEWithLogitsLoss for binary classification (CVD, liver)
- Weighted CrossEntropyLoss for 3-class kidney classification
- Per-component task weights for metabolic syndrome
- Masked loss calculation to handle NaN targets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import onnx  # For ONNX export verification

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


# ============================================================
# UNCERTAINTY WEIGHTING (Kendall et al., 2018)
# ============================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    Learns task-specific variance parameters to dynamically balance losses.
    
    Formula: L_total = sum( 1/(2*sigma^2) * L_i + log(sigma) )
    Using log_vars for numerical stability: s = log(sigma^2)
    Rewritten: L = exp(-s) * L_i + s/2
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
               Kendall, Gal & Cipolla (CVPR 2018)
    """
    def __init__(self, num_tasks):
        super(UncertaintyWeightedLoss, self).__init__()
        # Learnable log-variance for each task (initialized to 0 = sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, input_losses):
        """
        Args:
            input_losses: List of task losses [loss_cvd, loss_metabolic, loss_kidney, loss_liver]
        
        Returns:
            Weighted total loss (scalar)
        """
        total_loss = 0
        for i, loss in enumerate(input_losses):
            # precision = 1/sigma^2 = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            # L = precision * loss + log(sigma) = precision * loss + log_var/2
            total_loss = total_loss + (precision * loss + 0.5 * self.log_vars[i])
        
        return total_loss
    
    def get_weights(self):
        """Return current task weights (1/sigma^2) for logging."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars).cpu().numpy()
        return weights


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def masked_focal_loss(pred, target, mask, device, alpha=1.0, gamma=2.0):
    """
    Focal Loss for binary classification - focuses on hard examples.
    
    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive class
    FL(p) = -(1-alpha) * p^gamma * log(1-p)  for negative class
    
    Args:
        pred: Model predictions (logits) [batch, n]
        target: Ground truth labels [batch, n]
        mask: Valid target mask (1=valid, 0=was NaN) [batch, n]
        device: torch device
        alpha: Balance factor for positive class (default: 1.0)
        gamma: Focusing parameter - higher = more focus on hard examples (default: 2.0)
    
    Returns:
        Mean focal loss over valid targets only
    """
    # Convert logits to probabilities
    p = torch.sigmoid(pred)
    
    # Compute focal weight: (1-p)^gamma for positives, p^gamma for negatives
    p_t = p * target + (1 - p) * (1 - target)  # p if y=1, 1-p if y=0
    focal_weight = (1 - p_t) ** gamma
    
    # Standard BCE
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Apply focal weight and alpha
    focal_loss = focal_weight * bce
    
    # Apply mask
    masked_loss = focal_loss * mask
    
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    return torch.tensor(0.0, device=device)


def masked_weighted_bce_loss(pred, target, mask, device, pos_weight=None):
    """
    Weighted Binary Cross Entropy loss with masking for NaN targets.
    (Kept for metabolic components that need pos_weight)
    """
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    loss = loss_fn(pred, target)
    masked_loss = loss * mask
    
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    return torch.tensor(0.0, device=device)


def ordinal_weighted_bce_loss(pred, target, mask, device):
    """
    Weighted BCE for ordinal binary decomposition (kidney head).
    
    Applies different pos_weights to each node:
      - Node A (ACR >= 30): pos_weight = 4.5 (~18% positive)
      - Node B (ACR >= 300): pos_weight = 30.0 (~3% positive)
    
    This penalizes missing disease signals more heavily.
    
    Args:
        pred: Model predictions (logits) [batch, 2]
        target: Ground truth ordinal encoding [batch, 2]
        mask: Valid target mask [batch, 2]
        device: torch device
    
    Returns:
        Mean weighted loss over valid targets
    """
    # Per-node weights: [Node A, Node B]
    ordinal_weights = torch.tensor([4.5, 30.0], device=device, dtype=torch.float32)
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=ordinal_weights)
    loss = loss_fn(pred, target)
    
    # Apply mask
    masked_loss = loss * mask
    
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    return torch.tensor(0.0, device=device)


def masked_ce_loss(pred, target, device, class_weights=None):
    """
    CrossEntropyLoss for 3-class kidney classification.
    Uses ignore_index=-1 to mask NaN targets automatically.
    
    Args:
        pred: Model predictions (logits) [batch, 3]
        target: Ground truth labels [batch] (dtype=long, -1 for masked)
        device: torch device (cuda or cpu)
        class_weights: Optional weights for class imbalance [3]
    
    Returns:
        Mean loss over valid targets only
    """
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='mean')
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    
    return loss_fn(pred, target)


def compute_metabolic_loss(pred, target, mask, device):
    """
    Compute weighted loss for 5 metabolic syndrome components.
    Each component has its own pos_weight and task_weight.
    
    Args:
        pred: Model predictions [batch, 5]
        target: Ground truth labels [batch, 5]
        mask: Valid target mask [batch, 5]
        device: torch device
    
    Returns:
        Weighted sum of individual component losses
    """
    # Component order: [Waist, Triglycerides, HDL, BP, Glucose]
    pos_weights = [
        config.CLASS_WEIGHTS.get('waist_pos_weight'),       # None = balanced
        config.CLASS_WEIGHTS.get('triglycerides_pos_weight'),# 3.76
        config.CLASS_WEIGHTS.get('hdl_pos_weight'),         # 2.40
        config.CLASS_WEIGHTS.get('bp_pos_weight'),          # 1.78
        config.CLASS_WEIGHTS.get('glucose_pos_weight')      # None = balanced
    ]
    
    task_weights = [
        config.TASK_WEIGHTS.get('metabolic_waist', 1.0),    # 1.0
        config.TASK_WEIGHTS.get('metabolic_trig', 0.5),     # 0.5 (58% missing)
        config.TASK_WEIGHTS.get('metabolic_hdl', 1.0),      # 1.0
        config.TASK_WEIGHTS.get('metabolic_bp', 1.0),       # 1.0
        config.TASK_WEIGHTS.get('metabolic_glucose', 0.5)   # 0.5 (60% missing)
    ]
    
    total_loss = torch.tensor(0.0, device=device)
    
    for i in range(5):
        component_pred = pred[:, i:i+1]
        component_target = target[:, i:i+1]
        component_mask = mask[:, i:i+1]
        
        loss_i = masked_weighted_bce_loss(
            component_pred, component_target, component_mask, 
            device, pos_weight=pos_weights[i]
        )
        
        total_loss = total_loss + (task_weights[i] * loss_i)
    
    return total_loss


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
    
    # Optimize DataLoader for GPU training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # Larger batch for GPU efficiency
        shuffle=True,
        drop_last=True,
        num_workers=0,   # Must be 0 on Windows due to pickle issues
        pin_memory=True  # Faster CPU->GPU transfer
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ============================================================
    # 2. INITIALIZE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("Initializing Model...")
    print("=" * 60)
    
    num_continuous = len(config.CONT_COLS)
    print(f"Number of input features: {num_continuous}")
    
    mtl_model = SharedBottomMTL(
        num_continuous=num_continuous,
        hidden_dim=config.HIDDEN_DIM
    )
    
    # Move model to device
    mtl_model = mtl_model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in mtl_model.parameters())
    trainable_params = sum(p.numel() for p in mtl_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print task configuration
    print("\nTask Configuration:")
    print(f"  CVD pos_weight: {config.CLASS_WEIGHTS.get('cvd_pos_weight')}")
    print(f"  Liver pos_weight: {config.CLASS_WEIGHTS.get('liver_pos_weight')}")
    print(f"  Kidney class_weights: {config.CLASS_WEIGHTS.get('kidney_class_weights')}")
    
    # ============================================================
    # 3. SETUP OPTIMIZATION
    # ============================================================
    
    # Uncertainty Weighting (Kendall et al., 2018)
    # Combined with threshold optimization for production calibration
    mtl_loss_wrapper = UncertaintyWeightedLoss(num_tasks=4).to(device)
    
    # Include loss wrapper parameters in optimizer
    optimizer = optim.Adam(
        list(mtl_model.parameters()) + list(mtl_loss_wrapper.parameters()), 
        lr=config.LEARNING_RATE,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 0)
    )
    
    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # ============================================================
    # 4. TRAINING LOOP
    # ============================================================
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(config.EPOCHS):
        mtl_model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Track individual task losses
        loss_cardio_sum = 0.0
        loss_metabolic_sum = 0.0
        loss_kidney_sum = 0.0
        loss_liver_sum = 0.0
        
        for batch in train_loader:
            # Unpack batch (includes masks)
            x_cont, y_cardio, y_metabolic, y_kidney, y_liver, \
            mask_cardio, mask_metabolic, mask_kidney, mask_liver = batch
            
            # Move data to device
            x_cont = x_cont.to(device)
            y_cardio = y_cardio.to(device)
            y_metabolic = y_metabolic.to(device)
            y_kidney = y_kidney.to(device)  # Now dtype=long
            y_liver = y_liver.to(device)
            mask_cardio = mask_cardio.to(device)
            mask_metabolic = mask_metabolic.to(device)
            mask_kidney = mask_kidney.to(device)
            mask_liver = mask_liver.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            p_cardio, p_metabolic, p_kidney, p_liver = mtl_model(x_cont)
            
            # ============================================================
            # CALCULATE WEIGHTED MASKED LOSSES
            # ============================================================
            
            # Task A: CVD - Focal Loss (better than weighted BCE for extreme imbalance)
            loss_c = masked_focal_loss(p_cardio, y_cardio, mask_cardio, device, gamma=2.0)
            
            # Task B: Metabolic Syndrome - Per-component weighted BCE
            loss_m = compute_metabolic_loss(p_metabolic, y_metabolic, mask_metabolic, device)
            
            # Task C: Kidney - Weighted Ordinal Binary Decomposition
            # Node A (ACR>=30): pos_weight=4.5, Node B (ACR>=300): pos_weight=30.0
            loss_k = ordinal_weighted_bce_loss(p_kidney, y_kidney, mask_kidney, device)
            
            # Task D: Liver - Focal Loss (better for imbalanced classification)
            loss_l = masked_focal_loss(p_liver, y_liver, mask_liver, device, gamma=2.0)
            
            # Dynamic Task Weighting via Uncertainty
            # Use threshold optimization for production calibration
            loss = mtl_loss_wrapper([loss_c, loss_m, loss_k, loss_l])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability (prevents explosion from high Node B weight)
            torch.nn.utils.clip_grad_norm_(mtl_model.parameters(), max_norm=1.0)
            
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
        
        # Log learned task weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            weights = mtl_loss_wrapper.get_weights()
            print(f"  --> Learned weights: CVD={weights[0]:.3f}, Met={weights[1]:.3f}, "
                  f"Kid={weights[2]:.3f}, Liv={weights[3]:.3f}")
        
        # Step the scheduler (CosineAnnealing uses epoch)
        scheduler.step()
    
    # ============================================================
    # 5. SAVE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Save the trained model (PyTorch format)
    model_path = _package_dir / "trained_model.pth"
    torch.save(mtl_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # ============================================================
    # 6. EXPORT TO ONNX
    # ============================================================
    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    print("=" * 60)
    
    # Set model to eval mode for export
    mtl_model.eval()
    
    # Move to CPU for ONNX export (more compatible)
    mtl_model_cpu = mtl_model.cpu()
    
    # Create dummy input with correct shape (on CPU)
    dummy_input = torch.randn(1, num_continuous)
    
    # Define output names for the 4 task heads
    output_names = ['cardio_logits', 'metabolic_logits', 'kidney_logits', 'liver_logits']
    
    # Export to ONNX using legacy exporter
    onnx_path = _package_dir / "trained_model.onnx"
    
    try:
        torch.onnx.export(
            mtl_model_cpu,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=output_names,
            dynamic_axes={
                'input_features': {0: 'batch_size'},
                'cardio_logits': {0: 'batch_size'},
                'metabolic_logits': {0: 'batch_size'},
                'kidney_logits': {0: 'batch_size'},
                'liver_logits': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"ONNX model saved to: {onnx_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: OK - Valid")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Model saved as .pth only. Install 'onnxscript' for ONNX export.")


if __name__ == "__main__":
    train()

