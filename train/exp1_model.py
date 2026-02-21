import torch.nn as nn
import torch.nn.functional as F
import torch


class TwoScaleConv(nn.Module):
    """Two parallel convolutions with different kernel sizes for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size1=25, kernel_size2=9):
        super().__init__()
        # Each scale produces half the output channels
        self.conv1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=kernel_size1, stride=1, padding=kernel_size1//2)
        self.conv2 = nn.Conv1d(in_channels, out_channels//2, kernel_size=kernel_size2, stride=1, padding=kernel_size2//2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        return torch.cat([x1, x2], dim=1)


class CNNBranch(nn.Module):
    """CNN branch for processing ECG or PPG signal"""
    def __init__(self):
        super().__init__()
        # Initial expansion: 1024×1 -> 1024×32
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.ReLU()
        )
        
        # First two-scale conv block: 1024×32 -> 1024×32
        self.conv_block1 = TwoScaleConv(32, 32, kernel_size1=25, kernel_size2=9)
        
        # Downsample to 256: 1024×32 -> 256×32
        self.downsample1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Max pooling: 256×32 -> 128×32
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second two-scale conv block: 128×32 -> 128×32
        self.conv_block2 = TwoScaleConv(32, 32, kernel_size1=25, kernel_size2=9)
        
        # Downsample to 64: 128×32 -> 64×32
        self.downsample2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.initial_conv(x)  # 1024×32
        x = self.conv_block1(x)   # 1024×32
        x = self.downsample1(x)   # 256×32
        x = self.maxpool(x)       # 128×32
        x = self.conv_block2(x)   # 128×32
        x = self.downsample2(x)   # 64×32
        x = self.dropout(x)
        return x  # (batch, 32, 64)


class LSCN(nn.Module):
    """
    LSTM-CNN model for blood pressure estimation from ECG and PPG.
    
    Architecture:
    - Input: ECG (1024×1) and PPG (1024×1)
    - CNN Layer: Two parallel branches for ECG and PPG
      - Each branch: 2-Scale-Conv (k=25,9) blocks with downsampling
      - Output: 64×32 features per branch
    - Stack: Concatenate ECG and PPG features -> 64×64
    - LSTM Layer: Process temporal features -> 128 hidden units
    - Dropout: 0.3 for regularization
    - Output Layer: Dense layer -> SBP and DBP (2 values)
    """
    def __init__(self):
        super().__init__()
        # CNN branches for ECG and PPG
        self.ecg_branch = CNNBranch()
        self.ppg_branch = CNNBranch()
        
        # LSTM layer: input 64 features, hidden size 128
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Output layer: dense layer for SBP and DBP
        self.output_layer = nn.Linear(128, 2)
    
    def forward(self, ecg, ppg):
        """
        Forward pass.
        
        Args:
            ecg: (batch_size, 1, 1024) - ECG signal
            ppg: (batch_size, 1, 1024) - PPG signal
            
        Returns:
            (batch_size, 2) - Predicted SBP and DBP
        """
        # Process ECG and PPG through their respective CNN branches
        ecg_features = self.ecg_branch(ecg)  # (batch, 32, 64)
        ppg_features = self.ppg_branch(ppg)  # (batch, 32, 64)
        
        # Stack features: concatenate along channel dimension
        combined = torch.cat([ecg_features, ppg_features], dim=1)  # (batch, 64, 64)
        
        # Reshape for LSTM: (batch, seq_len, features)
        combined = combined.permute(0, 2, 1)  # (batch, 64, 64)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # (batch, 128)
        
        # Dropout on last hidden state
        last_hidden = self.dropout(last_hidden)

        # Output layer
        output = self.output_layer(last_hidden)  # (batch, 2) - SBP and DBP
        
        return output
