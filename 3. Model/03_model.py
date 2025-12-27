# model.py
"""
SharedBottomMTL: Multi-Task Learning neural network with hard parameter sharing.
This defines the architecture with a shared backbone and 4 task-specific heads.
"""

import torch
import torch.nn as nn


class SharedBottomMTL(nn.Module):
    """
    Shared-Bottom Multi-Task Learning Model.
    
    Architecture:
    - Input layer with BatchNorm for continuous features
    - Shared encoder backbone (dynamically sized based on hidden_dim)
    - Four task-specific heads:
        - Cardio: Binary classification (1 output)
        - Metabolic: Multi-label classification (5 outputs)
        - Kidney: Ordinal Binary Decomposition (2 outputs)
        - Liver: Binary classification (1 output)
    """
    
    def __init__(self, num_continuous, hidden_dim=256, dropout_rate=0.2):
        """
        Args:
            num_continuous: Number of continuous input features
            hidden_dim: Dimension of the final shared layer (default: 256)
            dropout_rate: Dropout probability for regularization (default: 0.2)
        """
        super(SharedBottomMTL, self).__init__()
        
        # Store config for reference
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # 1. Input Processing
        # BatchNorm for continuous inputs (data is already scaled, but helps training)
        self.input_bn = nn.BatchNorm1d(num_continuous)

        # 2. Shared Encoder Backbone (Hard Parameter Sharing)
        # Dynamic sizing: wider backbone for larger hidden_dim to maintain capacity
        # Layer sizes scale proportionally: 2x -> 1.5x -> 1x hidden_dim
        layer1_dim = min(hidden_dim * 2, 2048)  # Cap at 2048 for memory
        layer2_dim = min(int(hidden_dim * 1.5), 1536)  # Cap at 1536
        
        self.shared_backbone = nn.Sequential(
            # Layer 1: Input -> layer1_dim
            nn.Linear(num_continuous, layer1_dim),
            nn.BatchNorm1d(layer1_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Layer 2: layer1_dim -> layer2_dim
            nn.Linear(layer1_dim, layer2_dim),
            nn.BatchNorm1d(layer2_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Layer 3: layer2_dim -> hidden_dim
            nn.Linear(layer2_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        # 3. Task-Specific Heads
        # Head A: Cardiovascular - Binary Classification
        self.head_cardio = nn.Linear(hidden_dim, 1)
        
        # Head B: Metabolic Syndrome - Multi-Label Classification (5 labels)
        self.head_metabolic = nn.Linear(hidden_dim, 5)
        
        # Head C: Kidney Function - Ordinal Binary Decomposition
        # Node A: Is ACR >= 30? (At least Micro)
        # Node B: Is ACR >= 300? (Macro)
        # Encoding: Normal=[0,0], Micro=[1,0], Macro=[1,1]
        self.head_kidney = nn.Linear(hidden_dim, 2)
        
        # Head D: Liver Function - Binary Classification
        self.head_liver = nn.Linear(hidden_dim, 1)

    def forward(self, x_cont):
        """
        Forward pass through the network.
        
        Args:
            x_cont: Continuous input features tensor [batch_size, num_continuous]
            
        Returns:
            Tuple of 4 outputs (cardio, metabolic, kidney, liver)
        """
        # Normalize inputs
        x = self.input_bn(x_cont)
        
        # Pass through shared backbone
        z = self.shared_backbone(x)
        
        # Get predictions from each task head
        out_cardio = self.head_cardio(z)      # [batch, 1]
        out_metabolic = self.head_metabolic(z) # [batch, 5]
        out_kidney = self.head_kidney(z)       # [batch, 2] - ordinal nodes
        out_liver = self.head_liver(z)         # [batch, 1]
        
        return out_cardio, out_metabolic, out_kidney, out_liver
