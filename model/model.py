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
    - Shared encoder backbone (256 -> 192 -> 128)
    - Four task-specific heads:
        - Cardio: Binary classification (1 output)
        - Metabolic: Multi-label classification (5 outputs)
        - Kidney: Regression (1 output)
        - Liver: Regression (1 output)
    """
    
    def __init__(self, num_continuous, hidden_dim=128):
        """
        Args:
            num_continuous: Number of continuous input features
            hidden_dim: Dimension of the final shared layer (default: 128)
        """
        super(SharedBottomMTL, self).__init__()

        # 1. Input Processing
        # BatchNorm for continuous inputs (data is already scaled, but helps training)
        self.input_bn = nn.BatchNorm1d(num_continuous)

        # 2. Shared Encoder Backbone (Hard Parameter Sharing)
        self.shared_backbone = nn.Sequential(
            # Layer 1: Input -> 256
            nn.Linear(num_continuous, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Layer 2: 256 -> 192
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Layer 3: 192 -> hidden_dim (128)
            nn.Linear(192, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        # 3. Task-Specific Heads
        # Head A: Cardiovascular - Binary Classification
        self.head_cardio = nn.Linear(hidden_dim, 1)
        
        # Head B: Metabolic Syndrome - Multi-Label Classification (5 labels)
        self.head_metabolic = nn.Linear(hidden_dim, 5)
        
        # Head C: Kidney Function - Regression (ACR_Log)
        self.head_kidney = nn.Linear(hidden_dim, 1)
        
        # Head D: Liver Function - Regression (ALT_Log)
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
        out_kidney = self.head_kidney(z)       # [batch, 1]
        out_liver = self.head_liver(z)         # [batch, 1]
        
        return out_cardio, out_metabolic, out_kidney, out_liver
