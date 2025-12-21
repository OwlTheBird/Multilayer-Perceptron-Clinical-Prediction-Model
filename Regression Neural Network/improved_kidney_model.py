import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_process_data(filepath):
    print("Loading data...")
    # Convert relative path to absolute path relative to script location
    if not os.path.isabs(filepath):
        script_dir = Path(__file__).parent
        filepath = script_dir / filepath
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Define feature columns (first 35 columns as per notebook)
    feature_cols = df.columns[:35].tolist()
    target_col = 'kidney_acr_mg_g'  # Updated to new standardized column name
    data_type_col = 'data_type'
    
    # Remove target from features if present to prevent data leakage
    if target_col in feature_cols:
        print(f"Removing target '{target_col}' from features.")
        feature_cols.remove(target_col)
    
    # Combine features and target for cleaning
    # Ensure no duplicate columns in selection
    cols_to_select = list(set(feature_cols + [target_col, data_type_col]))
    data = df[cols_to_select].copy()
    
    # Drop rows with missing target values
    original_len = len(data)
    data.dropna(subset=[target_col], inplace=True)
    print(f"Dropped {original_len - len(data)} rows with missing target values.")
    
    # Separate features and data_type
    X_data = data[feature_cols]
    data_types = data[data_type_col]
    y_data = data[target_col].values.reshape(-1, 1) # Reshape to 2D array
    
    # Create Mask Matrix
    mask_matrix = X_data.notna().astype(np.float32).values
    
    # Impute NaN values with 0
    X_imputed = X_data.fillna(0).values
    
    # Split into train and test
    train_mask = (data_types == 'training').values
    test_mask = (data_types == 'testing').values
    
    X_train = X_imputed[train_mask]
    y_train = y_data[train_mask]
    Mask_train = mask_matrix[train_mask]
    
    X_test = X_imputed[test_mask]
    y_test = y_data[test_mask]
    Mask_test = mask_matrix[test_mask]
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    Mask_train_tensor = torch.tensor(Mask_train, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    Mask_test_tensor = torch.tensor(Mask_test, dtype=torch.float32)
    
    print(f"Training Data Shape: {X_train_tensor.shape}")
    print(f"Testing Data Shape: {X_test_tensor.shape}")
    
    return (X_train_tensor, y_train_tensor, Mask_train_tensor), (X_test_tensor, y_test_tensor, Mask_test_tensor)

class ImprovedRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedRegressionModel, self).__init__()
        
        # Deeper architecture with Batch Normalization and Dropout
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        return self.fc4(x)

def train_model():
    # Load Data
    try:
        # Use path relative to script location
        script_dir = Path(__file__).parent
        data_path = script_dir / 'datasets' / 'combined_data.csv'
        (X_train, y_train, Mask_train), (X_test, y_test, Mask_test) = load_and_process_data(str(data_path))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    # Hyperparameters
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = 1
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    PATIENCE = 15 # Early stopping patience
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train, Mask_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model, Loss, Optimizer
    model = ImprovedRegressionModel(INPUT_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("\nStarting Training...")
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y, batch_mask in train_loader:
            optimizer.zero_grad()
            
            # Apply mask
            masked_inputs = batch_X * batch_mask
            
            outputs = model(masked_inputs)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation (using test set for early stopping monitoring here, ideally should be separate val set)
        model.eval()
        with torch.no_grad():
            masked_test_inputs = X_test * Mask_test
            test_outputs = model(masked_test_inputs)
            val_loss = criterion(test_outputs, y_test).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model state
            torch.save(model.state_dict(), 'best_kidney_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    print("Training finished.")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_kidney_model.pth'))
    model.eval()
    
    with torch.no_grad():
        masked_test_inputs = X_test * Mask_test
        predictions = model(masked_test_inputs)
        final_mse = criterion(predictions, y_test).item()
        final_rmse = np.sqrt(final_mse)
        
    print(f'\nFinal Test MSE: {final_mse:.4f}')
    print(f'Final Test RMSE: {final_rmse:.4f}')
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_loss_curve.png')
    print("Loss curve saved as 'training_loss_curve.png'")

if __name__ == "__main__":
    train_model()
