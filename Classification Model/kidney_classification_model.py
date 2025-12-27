"""
Kidney Disease Classification Model
====================================
A 3-class neural network classifier for predicting kidney disease stages:
- Class 0: Normal (ACR < 30 mg/g)
- Class 1: Microalbuminuria (30 ≤ ACR < 300 mg/g)
- Class 2: Macroalbuminuria (ACR ≥ 300 mg/g)

Based on KDIGO clinical guidelines for albuminuria staging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CLINICAL THRESHOLDS (KDIGO Guidelines)
# ============================================================================
ACR_MICRO_THRESHOLD = 30    # mg/g - Microalbuminuria cutoff
ACR_MACRO_THRESHOLD = 300   # mg/g - Macroalbuminuria cutoff

CLASS_NAMES = ['Normal', 'Microalbuminuria', 'Macroalbuminuria']


def acr_to_class(acr_log_value):
    """
    Convert log-transformed ACR value back to original scale and classify.
    The dataset uses log-transformed values, so we need to exp() them first.
    
    Args:
        acr_log_value: Log-transformed ACR value
        
    Returns:
        Class label: 0 (Normal), 1 (Micro), 2 (Macro)
    """
    # Convert from log scale back to original
    acr_original = np.exp(acr_log_value)
    
    if acr_original < ACR_MICRO_THRESHOLD:
        return 0  # Normal
    elif acr_original < ACR_MACRO_THRESHOLD:
        return 1  # Microalbuminuria
    else:
        return 2  # Macroalbuminuria


def load_and_process_data(filepath='datasets/combined_data.csv'):
    """
    Load data from combined CSV file with pre-computed classification labels.
    
    Args:
        filepath: Path to combined_data.csv with data_type column
        
    Returns:
        Training and testing data tensors with class labels
    """
    print("Loading data from CSV file...")
    
    # Handle path resolution
    if not os.path.isabs(filepath):
        script_dir = Path(__file__).parent
        filepath = script_dir / filepath
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load CSV file
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} total samples")
    
    # Target column - albuminuria_risk is the kidney classification target
    # 0 = Normal, 1 = Microalbuminuria, 2 = Macroalbuminuria
    target_col = 'albuminuria_risk'
    data_type_col = 'data_type'
    
    # Define columns to exclude (targets and non-feature columns)
    exclude_cols = [
        # Classification targets
        'albuminuria_risk', 'liver_dysfunction',
        'has_cardiovascular_disease', 'high_waist_circumference',
        'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 
        'high_blood_pressure', 'high_glucose_mg_dl',
        # Metadata
        'data_type'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    # Filter rows with valid target values
    original_len = len(df)
    df = df.dropna(subset=[target_col])
    print(f"Dropped {original_len - len(df)} rows with missing target values")
    
    # Check class distribution
    print("\nClass distribution:")
    class_counts = df[target_col].value_counts().sort_index()
    for cls_idx, count in class_counts.items():
        cls_idx = int(cls_idx)
        print(f"  {CLASS_NAMES[cls_idx]}: {count} ({count/len(df)*100:.1f}%)")
    
    # Extract features
    X_data = df[feature_cols].fillna(0).values.astype(np.float32)
    y_data = df[target_col].values.astype(np.int64)
    data_types = df[data_type_col].values
    
    # Split by data_type column
    train_mask = data_types == 'training'
    test_mask = data_types == 'testing'
    
    X_train = X_data[train_mask]
    y_train = y_data[train_mask]
    
    X_test = X_data[test_mask]
    y_test = y_data[test_mask]
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    
    print(f"\nTraining set: {X_train_tensor.shape[0]} samples")
    print(f"Testing set: {X_test_tensor.shape[0]} samples")
    print(f"Feature dimension: {X_train_tensor.shape[1]}")
    
    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), feature_cols


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class KidneyClassificationMLP(nn.Module):
    """
    Multi-layer Perceptron for 3-class kidney disease classification.
    
    Architecture:
        Input → Linear(128) + BN + ReLU + Dropout
              → Linear(64) + BN + ReLU + Dropout  
              → Linear(32) + BN + ReLU + Dropout
              → Linear(3) → Softmax
    """
    
    def __init__(self, input_size, num_classes=3, dropout_rate=0.3):
        super(KidneyClassificationMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate * 0.67)  # Less dropout in final layer
        
        self.fc4 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)  # Raw logits (CrossEntropyLoss expects logits)
        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def compute_class_weights(y_train):
    """Compute class weights for imbalanced data."""
    class_counts = np.bincount(y_train.numpy())
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_model():
    """
    Train the kidney classification model using CSV data files.
    """
    print("=" * 60) 
    print("KIDNEY DISEASE CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Load data from CSV files
    try:
        (X_train, y_train), (X_test, y_test), feature_cols = load_and_process_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Hyperparameters
    INPUT_SIZE = X_train.shape[1]
    NUM_CLASSES = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    PATIENCE = 15
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Input features: {INPUT_SIZE}")
    print(f"Output classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(y_train)
    print(f"\nClass weights: {class_weights.numpy()}")
    
    # Model, Loss, Optimizer
    model = KidneyClassificationMLP(INPUT_SIZE, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)
            
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_test.numpy(), val_preds.numpy())
            val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_kidney_classification_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    print("\nTraining complete!")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_kidney_classification_model.pth'))
    model.eval()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.softmax(test_outputs, dim=1)
        _, test_preds = torch.max(test_outputs, 1)
    
    y_true = y_test.numpy()
    y_pred = test_preds.numpy()
    y_probs = test_probs.numpy()
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall:    {recall:.4f} (weighted)")
    print(f"  F1-Score:  {f1:.4f} (weighted)")
    
    # Per-class AUC-ROC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        auc_scores = []
        for i in range(NUM_CLASSES):
            if y_true_bin[:, i].sum() > 0:  # Only compute if class exists
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                auc_scores.append(auc)
                print(f"  AUC-ROC ({CLASS_NAMES[i]}): {auc:.4f}")
            else:
                print(f"  AUC-ROC ({CLASS_NAMES[i]}): N/A (no samples)")
        print(f"  AUC-ROC (macro avg): {np.mean(auc_scores):.4f}")
    except Exception as e:
        print(f"  AUC-ROC: Could not compute ({e})")
    
    # Classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Generate visualizations
    generate_plots(train_losses, val_losses, val_accuracies, y_true, y_pred, y_probs)
    
    # Export to ONNX format
    export_to_onnx(model, INPUT_SIZE)
    
    print(f"\n{'='*60}")
    print("FILES SAVED")
    print("=" * 60)
    print("  - best_kidney_classification_model.pth")
    print("  - kidney_classification_model.onnx")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png")
    
    return model


def export_to_onnx(model, input_size, filename='kidney_classification_model.onnx'):
    """
    Export the trained model to ONNX format for production deployment.
    
    Args:
        model: Trained PyTorch model
        input_size: Number of input features
        filename: Output ONNX file name
    """
    print(f"\n{'='*60}")
    print("EXPORTING TO ONNX")
    print("=" * 60)
    
    try:
        model.eval()
        
        # Create dummy input with correct shape
        dummy_input = torch.randn(1, input_size, requires_grad=False)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"  Model exported to: {filename}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model)
            print("  ONNX model validation: PASSED")
        except ImportError:
            print("  Note: Install 'onnx' package to validate the exported model")
        except Exception as e:
            print(f"  ONNX validation warning: {e}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print("  Note: Install 'onnxscript' package for ONNX export with newer PyTorch")


def generate_plots(train_losses, val_losses, val_accuracies, y_true, y_pred, y_probs):
    """Generate training curves, confusion matrix, and ROC curves."""
    
    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Kidney Disease Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    
    # 3. ROC curves
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green']
    
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if y_true_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Kidney Disease Classification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train_model()
