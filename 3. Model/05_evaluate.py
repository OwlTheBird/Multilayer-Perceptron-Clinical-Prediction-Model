# evaluate.py
"""
Evaluation script for the Multi-Task Learning model.
Evaluates the model on the test set and computes metrics for all 4 tasks.

Updated to include:
- 3-class classification metrics for kidney (Macro-F1, per-class precision/recall)
- Binary classification metrics for liver (previously regression)
- Precision-Recall AUC for imbalanced binary targets
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve
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


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_optimal_threshold(y_true, y_probs, metric='f2'):
    """
    Find optimal classification threshold for binary tasks.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities (0-1)
        metric: 'f2' (favors recall) or 'youden' (balanced)
    
    Returns:
        Optimal threshold value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    if metric == 'f2':
        # F2 Score: Weighs Recall 2x more than Precision
        f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        ix = np.argmax(f2_scores[:-1])  # Exclude last element (recall=0)
        best_thresh = thresholds[ix]
        best_score = f2_scores[ix]
    elif metric == 'youden':
        # Youden's J: Sensitivity + Specificity - 1
        # Using precision-recall curve as proxy
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        ix = np.argmax(f1_scores[:-1])
        best_thresh = thresholds[ix]
        best_score = f1_scores[ix]
    else:
        best_thresh = 0.5
        best_score = 0.0
    
    return best_thresh, best_score


def evaluate_task_binary(y_true, y_pred_logits, mask, task_name):
    """
    Evaluate binary classification task with advanced metrics.
    Includes threshold optimization for clinical sensitivity.
    """
    # Filter to valid predictions only
    valid_mask = mask.flatten().bool().numpy()
    if valid_mask.sum() == 0:
        return {}
    
    y_true_valid = y_true[valid_mask].flatten().numpy()
    y_pred_logits_valid = y_pred_logits[valid_mask].flatten()
    y_pred_proba = torch.sigmoid(y_pred_logits_valid).numpy()
    
    # Use configured optimal threshold (from config.OPTIMAL_THRESHOLDS)
    threshold = config.OPTIMAL_THRESHOLDS.get(task_name.lower(), 0.5)
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true_valid, y_pred_binary),
        'precision': precision_score(y_true_valid, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_valid, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true_valid, y_pred_binary, zero_division=0),
        'macro_f1': f1_score(y_true_valid, y_pred_binary, average='macro', zero_division=0),
    }
    
    # ROC-AUC and PR-AUC (if both classes present)
    if len(np.unique(y_true_valid)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_valid, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true_valid, y_pred_proba)
        except Exception:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(y_true_valid, y_pred_binary)
    metrics['n_samples'] = int(valid_mask.sum())
    metrics['positive_rate'] = float(y_true_valid.mean())
    
    return metrics


def evaluate_task_kidney(y_true, y_pred_logits, mask):
    """
    Evaluate kidney ordinal binary decomposition.
    
    Decoding:
        [0, 0] -> Normal (class 0)
        [1, 0] -> Microalbuminuria (class 1)
        [1, 1] -> Macroalbuminuria (class 2)
    
    Args:
        y_true: Ground truth ordinal encoding [N, 2]
        y_pred_logits: Model output logits [N, 2]
        mask: Valid target mask [N, 2]
    
    Returns:
        Dictionary with per-class and macro metrics
    """
    # Use first column of mask (both columns are same)
    valid_mask = mask[:, 0].bool().numpy()
    if valid_mask.sum() == 0:
        return {}
    
    y_true_valid = y_true[valid_mask].numpy()
    y_pred_logits_valid = y_pred_logits[valid_mask]
    
    # Convert logits to probabilities
    y_pred_proba = torch.sigmoid(y_pred_logits_valid).numpy()
    
    # Decode ordinal encoding to class labels
    # Node A (col 0): Is ACR >= 30?
    # Node B (col 1): Is ACR >= 300?
    def decode_ordinal(encoding):
        """Convert 2-node encoding to class: [0,0]->0, [1,0]->1, [1,1]->2"""
        node_a = (encoding[:, 0] > 0.5).astype(int)
        node_b = (encoding[:, 1] > 0.5).astype(int)
        # Class = Node A + Node B (since [1,1]->2, [1,0]->1, [0,0]->0)
        return node_a + node_b
    
    y_true_class = decode_ordinal(y_true_valid)
    y_pred_class = decode_ordinal(y_pred_proba)
    
    class_names = ['Normal', 'Microalbuminuria', 'Macroalbuminuria']
    
    metrics = {
        'accuracy': accuracy_score(y_true_class, y_pred_class),
        'macro_precision': precision_score(y_true_class, y_pred_class, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true_class, y_pred_class, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true_class, y_pred_class, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    for i, name in enumerate(class_names):
        class_mask = (y_true_class == i)
        if class_mask.sum() > 0:
            class_correct = (y_pred_class[class_mask] == i).mean()
            metrics[f'{name}_recall'] = float(class_correct)
            metrics[f'{name}_count'] = int(class_mask.sum())
    
    # Per-node metrics (for ordinal analysis)
    metrics['node_a_accuracy'] = float(((y_pred_proba[:, 0] > 0.5) == y_true_valid[:, 0]).mean())
    metrics['node_b_accuracy'] = float(((y_pred_proba[:, 1] > 0.5) == y_true_valid[:, 1]).mean())
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2])
    metrics['n_samples'] = int(valid_mask.sum())
    
    # Class distribution
    metrics['true_distribution'] = np.bincount(y_true_class, minlength=3).tolist()
    metrics['pred_distribution'] = np.bincount(y_pred_class, minlength=3).tolist()
    
    return metrics


def evaluate_task_metabolic(y_true, y_pred, mask):
    """Evaluate metabolic syndrome components (multi-label classification)."""
    valid_mask = mask.bool()
    if valid_mask.sum().item() == 0:
        return {}
    
    valid_mask_np = valid_mask.numpy()
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    y_pred_proba = torch.sigmoid(y_pred).numpy()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    labels = ['Waist', 'Triglycerides', 'HDL', 'Blood_Pressure', 'Glucose']
    metrics = {}
    
    # Per-label metrics
    for i, label in enumerate(labels):
        label_valid_mask = valid_mask_np[:, i]
        if label_valid_mask.sum() > 0:
            y_true_label = y_true_np[label_valid_mask, i]
            y_pred_label = y_pred_binary[label_valid_mask, i]
            y_proba_label = y_pred_proba[label_valid_mask, i]
            
            metrics[f'{label}_accuracy'] = accuracy_score(y_true_label, y_pred_label)
            metrics[f'{label}_precision'] = precision_score(y_true_label, y_pred_label, zero_division=0)
            metrics[f'{label}_recall'] = recall_score(y_true_label, y_pred_label, zero_division=0)
            metrics[f'{label}_f1'] = f1_score(y_true_label, y_pred_label, zero_division=0)
            metrics[f'{label}_macro_f1'] = f1_score(y_true_label, y_pred_label, average='macro', zero_division=0)
            
            if len(np.unique(y_true_label)) > 1:
                try:
                    metrics[f'{label}_roc_auc'] = roc_auc_score(y_true_label, y_proba_label)
                    metrics[f'{label}_pr_auc'] = average_precision_score(y_true_label, y_proba_label)
                except Exception:
                    metrics[f'{label}_roc_auc'] = 0.0
                    metrics[f'{label}_pr_auc'] = 0.0
            else:
                metrics[f'{label}_roc_auc'] = 0.0
                metrics[f'{label}_pr_auc'] = 0.0
                
            metrics[f'{label}_n_samples'] = int(label_valid_mask.sum())
    
    # Overall micro-averaged metrics
    valid_samples = valid_mask_np.flatten()
    y_true_flat = y_true_np.flatten()[valid_samples]
    y_pred_binary_flat = y_pred_binary.flatten()[valid_samples]
    
    if len(y_true_flat) > 0:
        metrics['micro_accuracy'] = accuracy_score(y_true_flat, y_pred_binary_flat)
        metrics['micro_f1'] = f1_score(y_true_flat, y_pred_binary_flat, zero_division=0)
    
    metrics['total_valid_samples'] = int(valid_mask_np.sum())
    
    return metrics


def evaluate():
    """Main evaluation function."""
    print("=" * 60)
    print("MODEL EVALUATION (Refactored)")
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
    mtl_model = SharedBottomMTL(
        num_continuous=num_continuous,
        hidden_dim=config.HIDDEN_DIM
    )
    
    model_path = _package_dir / "trained_model.pth"
    mtl_model.load_state_dict(torch.load(model_path, map_location=device))
    mtl_model = mtl_model.to(device)
    mtl_model.eval()
    
    # Collect all predictions and targets
    all_cardio_pred, all_cardio_true, all_cardio_mask = [], [], []
    all_metabolic_pred, all_metabolic_true, all_metabolic_mask = [], [], []
    all_kidney_pred, all_kidney_true, all_kidney_mask = [], [], []
    all_liver_pred, all_liver_true, all_liver_mask = [], [], []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            x_cont, y_cardio, y_metabolic, y_kidney, y_liver, \
            mask_cardio, mask_metabolic, mask_kidney, mask_liver = batch
            
            x_cont = x_cont.to(device)
            p_cardio, p_metabolic, p_kidney, p_liver = mtl_model(x_cont)
            
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
    
    # ============================================================
    # TASK A: CARDIOVASCULAR DISEASE (Binary Classification)
    # ============================================================
    print("\n" + "=" * 60)
    print("TASK A: CARDIOVASCULAR DISEASE (Binary Classification)")
    print("=" * 60)
    cardio_metrics = evaluate_task_binary(cardio_true, cardio_pred, cardio_mask, 'CVD')
    if cardio_metrics:
        print(f"Accuracy:    {cardio_metrics['accuracy']:.4f}")
        print(f"Precision:   {cardio_metrics['precision']:.4f}")
        print(f"Recall:      {cardio_metrics['recall']:.4f}")
        print(f"F1 Score:    {cardio_metrics['f1']:.4f}")
        print(f"Macro-F1:    {cardio_metrics['macro_f1']:.4f}")
        print(f"ROC-AUC:     {cardio_metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {cardio_metrics['pr_auc']:.4f}")
        print(f"Positive Rate: {cardio_metrics['positive_rate']:.2%}")
        print(f"Valid samples: {cardio_metrics['n_samples']}")
        print("\nConfusion Matrix:")
        print(cardio_metrics['confusion_matrix'])
    
    # ============================================================
    # TASK B: METABOLIC SYNDROME (Multi-Label Classification)
    # ============================================================
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
                print(f"  F1 Score:  {metabolic_metrics[f'{label}_f1']:.4f}")
                print(f"  Macro-F1:  {metabolic_metrics[f'{label}_macro_f1']:.4f}")
                print(f"  ROC-AUC:   {metabolic_metrics[f'{label}_roc_auc']:.4f}")
                print(f"  PR-AUC:    {metabolic_metrics[f'{label}_pr_auc']:.4f}")
                print(f"  Samples:   {metabolic_metrics[f'{label}_n_samples']}")
        
        print(f"\nOverall Micro-F1: {metabolic_metrics.get('micro_f1', 0):.4f}")
    
    # ============================================================
    # TASK C: KIDNEY FUNCTION (3-Class Ordinal Classification)
    # ============================================================
    print("\n" + "=" * 60)
    print("TASK C: KIDNEY FUNCTION (3-Class Ordinal Classification)")
    print("=" * 60)
    kidney_metrics = evaluate_task_kidney(kidney_true, kidney_pred, kidney_mask)
    if kidney_metrics:
        print(f"Accuracy:        {kidney_metrics['accuracy']:.4f}")
        print(f"Macro-Precision: {kidney_metrics['macro_precision']:.4f}")
        print(f"Macro-Recall:    {kidney_metrics['macro_recall']:.4f}")
        print(f"Macro-F1:        {kidney_metrics['macro_f1']:.4f}")
        print(f"Weighted-F1:     {kidney_metrics['weighted_f1']:.4f}")
        print(f"Valid samples:   {kidney_metrics['n_samples']}")
        
        print("\nPer-Class Recall:")
        for name in ['Normal', 'Microalbuminuria', 'Macroalbuminuria']:
            if f'{name}_recall' in kidney_metrics:
                print(f"  {name}: {kidney_metrics[f'{name}_recall']:.4f} (n={kidney_metrics[f'{name}_count']})")
        
        print("\nClass Distribution (True vs Predicted):")
        print(f"  True: {kidney_metrics['true_distribution']}")
        print(f"  Pred: {kidney_metrics['pred_distribution']}")
        
        print("\nConfusion Matrix:")
        print("         Pred:Normal  Pred:Micro  Pred:Macro")
        cm = kidney_metrics['confusion_matrix']
        for i, name in enumerate(['True:Normal', 'True:Micro', 'True:Macro']):
            print(f"  {name:12s} {cm[i][0]:6d}      {cm[i][1]:6d}      {cm[i][2]:6d}")
    
    # ============================================================
    # TASK D: LIVER FUNCTION (Binary Classification)
    # ============================================================
    print("\n" + "=" * 60)
    print("TASK D: LIVER FUNCTION (Binary Classification)")
    print("=" * 60)
    liver_metrics = evaluate_task_binary(liver_true, liver_pred, liver_mask, 'Liver')
    if liver_metrics:
        print(f"Accuracy:    {liver_metrics['accuracy']:.4f}")
        print(f"Precision:   {liver_metrics['precision']:.4f}")
        print(f"Recall:      {liver_metrics['recall']:.4f}")
        print(f"F1 Score:    {liver_metrics['f1']:.4f}")
        print(f"Macro-F1:    {liver_metrics['macro_f1']:.4f}")
        print(f"ROC-AUC:     {liver_metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {liver_metrics['pr_auc']:.4f}")
        print(f"Positive Rate: {liver_metrics['positive_rate']:.2%}")
        print(f"Valid samples: {liver_metrics['n_samples']}")
        print("\nConfusion Matrix:")
        print(liver_metrics['confusion_matrix'])
    
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
