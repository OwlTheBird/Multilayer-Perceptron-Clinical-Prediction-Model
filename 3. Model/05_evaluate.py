# evaluate.py
"""
Evaluation script for the Multi-Task Learning model.
Evaluates the model on the test set and computes metrics for all 4 tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Binary Cross Entropy loss that only considers valid (non-NaN) targets."""
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(pred, target)
    masked_loss = loss * mask
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=device)


def masked_mse_loss(pred, target, mask, device):
    """Mean Squared Error loss that only considers valid (non-NaN) targets."""
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    masked_loss = loss * mask
    valid_count = mask.sum()
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=device)


def evaluate_task_cardio(y_true, y_pred, mask):
    """Evaluate cardiovascular disease prediction (binary classification)."""
    # Filter to valid predictions only
    valid_mask = mask.flatten().bool().numpy()
    if valid_mask.sum() == 0:
        return {}
    
    y_true_valid = y_true[valid_mask].numpy()
    y_pred_valid = y_pred[valid_mask].numpy()
    y_pred_proba = torch.sigmoid(torch.tensor(y_pred_valid)).numpy()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true_valid, y_pred_binary),
        'precision': precision_score(y_true_valid, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_valid, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true_valid, y_pred_binary, zero_division=0),
    }
    
    # ROC-AUC (if both classes present)
    if len(np.unique(y_true_valid)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_valid, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(y_true_valid, y_pred_binary)
    metrics['n_samples'] = valid_mask.sum()
    
    return metrics


def evaluate_task_metabolic(y_true, y_pred, mask):
    """Evaluate metabolic syndrome components (multi-label classification)."""
    # Filter to valid predictions only
    valid_mask = mask.bool().numpy()
    if valid_mask.sum() == 0:
        return {}
    
    # Filter to valid samples (keep 2D shape)
    # valid_mask is (batch_size, 5) - we need to find rows where at least one label is valid
    row_valid = valid_mask.any(axis=1)
    y_true_valid = y_true[row_valid].numpy()  # Shape: (n_valid, 5)
    y_pred_valid = y_pred[row_valid].numpy()  # Shape: (n_valid, 5)
    
    y_pred_proba = torch.sigmoid(torch.tensor(y_pred_valid)).numpy()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    labels = ['Waist', 'Triglycerides', 'HDL', 'Blood_Pressure', 'Glucose']
    metrics = {}
    
    # Per-label metrics
    for i, label in enumerate(labels):
        # Get valid samples for this specific label
        label_valid_mask = valid_mask[row_valid, i]
        if label_valid_mask.sum() > 0:
            y_true_label = y_true_valid[label_valid_mask, i]
            y_pred_label = y_pred_binary[label_valid_mask, i]
            y_proba_label = y_pred_proba[label_valid_mask, i]
            
            metrics[f'{label}_accuracy'] = accuracy_score(y_true_label, y_pred_label)
            metrics[f'{label}_precision'] = precision_score(y_true_label, y_pred_label, zero_division=0)
            metrics[f'{label}_recall'] = recall_score(y_true_label, y_pred_label, zero_division=0)
            metrics[f'{label}_f1'] = f1_score(y_true_label, y_pred_label, zero_division=0)
            
            if len(np.unique(y_true_label)) > 1:
                try:
                    metrics[f'{label}_roc_auc'] = roc_auc_score(y_true_label, y_proba_label)
                except:
                    metrics[f'{label}_roc_auc'] = 0.0
            else:
                metrics[f'{label}_roc_auc'] = 0.0
    
    # Overall metrics (micro-averaged)
    metrics['micro_accuracy'] = accuracy_score(y_true_valid.flatten(), y_pred_binary.flatten())
    metrics['micro_precision'] = precision_score(y_true_valid.flatten(), y_pred_binary.flatten(), zero_division=0)
    metrics['micro_recall'] = recall_score(y_true_valid.flatten(), y_pred_binary.flatten(), zero_division=0)
    metrics['micro_f1'] = f1_score(y_true_valid.flatten(), y_pred_binary.flatten(), zero_division=0)
    
    metrics['n_samples'] = valid_mask.sum()
    
    return metrics


def evaluate_task_regression(y_true, y_pred, mask, task_name):
    """Evaluate regression tasks (kidney/liver)."""
    # Filter to valid predictions only
    valid_mask = mask.flatten().bool().numpy()
    if valid_mask.sum() == 0:
        return {}
    
    y_true_valid = y_true[valid_mask].numpy()
    y_pred_valid = y_pred[valid_mask].numpy()
    
    metrics = {
        'mse': mean_squared_error(y_true_valid, y_pred_valid),
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
        'r2': r2_score(y_true_valid, y_pred_valid),
        'mean_true': float(np.mean(y_true_valid)),
        'mean_pred': float(np.mean(y_pred_valid)),
        'n_samples': int(valid_mask.sum())
    }
    
    return metrics


def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate():
    """Main evaluation function."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = NHANESMultiTaskDataset(
        db_path=config.DB_PATH,
        table_name='testing_set',
        cont_cols=config.CONT_COLS,
        target_mapping=config.TARGET_MAPPING
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    num_continuous = len(config.CONT_COLS)
    model = SharedBottomMTL(
        num_continuous=num_continuous,
        hidden_dim=config.HIDDEN_DIM
    )
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Collect all predictions and targets
    all_cardio_pred = []
    all_cardio_true = []
    all_cardio_mask = []
    
    all_metabolic_pred = []
    all_metabolic_true = []
    all_metabolic_mask = []
    
    all_kidney_pred = []
    all_kidney_true = []
    all_kidney_mask = []
    
    all_liver_pred = []
    all_liver_true = []
    all_liver_mask = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            x_cont, y_cardio, y_metabolic, y_kidney, y_liver, \
            mask_cardio, mask_metabolic, mask_kidney, mask_liver = batch
            
            x_cont = x_cont.to(device)
            
            # Forward pass
            p_cardio, p_metabolic, p_kidney, p_liver = model(x_cont)
            
            # Collect predictions (move to CPU)
            all_cardio_pred.append(p_cardio.cpu())
            all_cardio_true.append(y_cardio.cpu())
            all_cardio_mask.append(mask_cardio.cpu())
            
            all_metabolic_pred.append(p_metabolic.cpu())
            all_metabolic_true.append(y_metabolic.cpu())
            all_metabolic_mask.append(mask_metabolic.cpu())
            
            all_kidney_pred.append(p_kidney.cpu())
            all_kidney_true.append(y_kidney.cpu())
            all_kidney_mask.append(mask_kidney.cpu())
            
            all_liver_pred.append(p_liver.cpu())
            all_liver_true.append(y_liver.cpu())
            all_liver_mask.append(mask_liver.cpu())
    
    # Concatenate all batches
    cardio_pred = torch.cat(all_cardio_pred, dim=0)
    cardio_true = torch.cat(all_cardio_true, dim=0)
    cardio_mask = torch.cat(all_cardio_mask, dim=0)
    
    metabolic_pred = torch.cat(all_metabolic_pred, dim=0)
    metabolic_true = torch.cat(all_metabolic_true, dim=0)
    metabolic_mask = torch.cat(all_metabolic_mask, dim=0)
    
    kidney_pred = torch.cat(all_kidney_pred, dim=0)
    kidney_true = torch.cat(all_kidney_true, dim=0)
    kidney_mask = torch.cat(all_kidney_mask, dim=0)
    
    liver_pred = torch.cat(all_liver_pred, dim=0)
    liver_true = torch.cat(all_liver_true, dim=0)
    liver_mask = torch.cat(all_liver_mask, dim=0)
    
    # Evaluate each task
    print("\n" + "=" * 60)
    print("TASK A: CARDIOVASCULAR DISEASE (Binary Classification)")
    print("=" * 60)
    cardio_metrics = evaluate_task_cardio(cardio_true, cardio_pred, cardio_mask)
    if cardio_metrics:
        print(f"Accuracy:  {cardio_metrics['accuracy']:.4f}")
        print(f"Precision: {cardio_metrics['precision']:.4f}")
        print(f"Recall:    {cardio_metrics['recall']:.4f}")
        print(f"F1 Score:  {cardio_metrics['f1']:.4f}")
        print(f"ROC-AUC:   {cardio_metrics['roc_auc']:.4f}")
        print(f"Valid samples: {cardio_metrics['n_samples']}")
        print("\nConfusion Matrix:")
        print(cardio_metrics['confusion_matrix'])
    
    print("\n" + "=" * 60)
    print("TASK B: METABOLIC SYNDROME (Multi-Label Classification)")
    print("=" * 60)
    metabolic_metrics = evaluate_task_metabolic(metabolic_true, metabolic_pred, metabolic_mask)
    if metabolic_metrics:
        labels = ['Waist', 'Triglycerides', 'HDL', 'Blood_Pressure', 'Glucose']
        for label in labels:
            if f'{label}_f1' in metabolic_metrics:
                print(f"\n{label}:")
                print(f"  Accuracy:  {metabolic_metrics[f'{label}_accuracy']:.4f}")
                print(f"  Precision: {metabolic_metrics[f'{label}_precision']:.4f}")
                print(f"  Recall:    {metabolic_metrics[f'{label}_recall']:.4f}")
                print(f"  F1 Score:  {metabolic_metrics[f'{label}_f1']:.4f}")
                print(f"  ROC-AUC:   {metabolic_metrics[f'{label}_roc_auc']:.4f}")
        
        print(f"\nOverall (Micro-averaged):")
        print(f"  Accuracy:  {metabolic_metrics['micro_accuracy']:.4f}")
        print(f"  Precision: {metabolic_metrics['micro_precision']:.4f}")
        print(f"  Recall:    {metabolic_metrics['micro_recall']:.4f}")
        print(f"  F1 Score:  {metabolic_metrics['micro_f1']:.4f}")
        print(f"  Valid samples: {metabolic_metrics['n_samples']}")
    
    print("\n" + "=" * 60)
    print("TASK C: KIDNEY FUNCTION (Regression - ACR Log)")
    print("=" * 60)
    kidney_metrics = evaluate_task_regression(kidney_true, kidney_pred, kidney_mask, 'kidney')
    if kidney_metrics:
        print(f"MSE:  {kidney_metrics['mse']:.4f}")
        print(f"RMSE: {kidney_metrics['rmse']:.4f}")
        print(f"MAE:  {kidney_metrics['mae']:.4f}")
        print(f"R²:   {kidney_metrics['r2']:.4f}")
        print(f"Mean True:  {kidney_metrics['mean_true']:.4f}")
        print(f"Mean Pred:  {kidney_metrics['mean_pred']:.4f}")
        print(f"Valid samples: {kidney_metrics['n_samples']}")
    
    print("\n" + "=" * 60)
    print("TASK D: LIVER FUNCTION (Regression - ALT Log)")
    print("=" * 60)
    liver_metrics = evaluate_task_regression(liver_true, liver_pred, liver_mask, 'liver')
    if liver_metrics:
        print(f"MSE:  {liver_metrics['mse']:.4f}")
        print(f"RMSE: {liver_metrics['rmse']:.4f}")
        print(f"MAE:  {liver_metrics['mae']:.4f}")
        print(f"R²:   {liver_metrics['r2']:.4f}")
        print(f"Mean True:  {liver_metrics['mean_true']:.4f}")
        print(f"Mean Pred:  {liver_metrics['mean_pred']:.4f}")
        print(f"Valid samples: {liver_metrics['n_samples']}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return {
        'cardio': cardio_metrics,
        'metabolic': metabolic_metrics,
        'kidney': kidney_metrics,
        'liver': liver_metrics
    }


if __name__ == "__main__":
    evaluate()


