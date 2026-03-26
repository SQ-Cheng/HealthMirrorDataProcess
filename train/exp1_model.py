import torch.nn as nn
import torch.nn.functional as F
import torch

CNN_DROPOUT = 0.5
OUTPUT_DROPOUT = 0.5

KERNEL_SIZE_1 = 25
KERNEL_SIZE_2 = 9

CNN_CHANNELS = 16
LSTM_HIDDEN_SIZE = 32
LSTM_INPUT_SIZE = CNN_CHANNELS * 4

class TwoScaleConv(nn.Module):
    """Two parallel convolutions with different kernel sizes for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size1=25, kernel_size2=9):
        super().__init__()
        # Each scale produces half the output channels
        self.conv1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=kernel_size1, stride=2, padding=kernel_size1//2)
        self.conv2 = nn.Conv1d(in_channels, out_channels//2, kernel_size=kernel_size2, stride=2, padding=kernel_size2//2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        return torch.cat([x1, x2], dim=1)


class CNNBranch(nn.Module):
    """CNN branch for processing ECG or PPG signal"""
    def __init__(self):
        super().__init__()
        
        # Input: 1024×1 -> Expand to 1024×32 for convolution
        self.expand = nn.Conv1d(1, CNN_CHANNELS, kernel_size=1)  # 1024×1 -> 1024×32

        self.relu = nn.ReLU()

        # Conv layers with kernel sizes 25 and 9
        self.conv_1_1 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_1, stride=1, padding=KERNEL_SIZE_1//2)  # 1024×32 -> 1024×32
        self.conv_2_1 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_2, stride=1, padding=KERNEL_SIZE_2//2)  # 1024×32 -> 1024×32
        # Max pooling layers for downsampling
        self.maxpool_1_1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 1024×32 -> 512×32
        self.maxpool_2_1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 1024×32 -> 512×32
        self.conv_1_2 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_1, stride=1, padding=KERNEL_SIZE_1//2)  # 512×32 -> 512×32
        self.conv_2_2 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_2, stride=1, padding=KERNEL_SIZE_2//2)  # 512×32 -> 512×32
        self.maxpool_1_2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 512×32 -> 256×32
        self.maxpool_2_2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 512×32 -> 256×32
        self.conv_1_3 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_1, stride=1, padding=KERNEL_SIZE_1//2)  # 256×32 -> 256×32
        self.conv_2_3 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_2, stride=1, padding=KERNEL_SIZE_2//2)  # 256×32 -> 256×32
        self.maxpool_1_3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 256×32 -> 128×32
        self.maxpool_2_3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 256×32 -> 128×32
        self.conv_1_4 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_1, stride=1, padding=KERNEL_SIZE_1//2)  # 128×32 -> 128×32
        self.conv_2_4 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=KERNEL_SIZE_2, stride=1, padding=KERNEL_SIZE_2//2)  # 128×32 -> 128×32
        self.maxpool_1_4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 128×32 -> 64×32
        self.maxpool_2_4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 128×32 -> 64×32

        # Dropout for regularization
        self.dropout = nn.Dropout(CNN_DROPOUT)
        
    def forward(self, x):
        x = self.relu(self.expand(x))  # (batch, 32, 1024)
        x1 = self.relu(self.conv_1_1(x))  # (batch, 32, 1024)
        x1 = self.maxpool_1_1(x1)  # (batch, 32, 512)
        x1 = self.relu(self.conv_1_2(x1))  # (batch, 32, 512)
        x1 = self.maxpool_1_2(x1)  # (batch, 32, 256)
        x1 = self.relu(self.conv_1_3(x1))  # (batch, 32, 256)
        x1 = self.maxpool_1_3(x1)  # (batch, 32, 128)
        x1 = self.relu(self.conv_1_4(x1))  # (batch, 32, 128)
        x1 = self.maxpool_1_4(x1)  # (batch, 32, 64)
        
        x2 = self.relu(self.conv_2_1(x))  # (batch, 32, 1024)
        x2 = self.maxpool_2_1(x2)  # (batch, 32, 512)
        x2 = self.relu(self.conv_2_2(x2))  # (batch, 32, 512)
        x2 = self.maxpool_2_2(x2)  # (batch, 32, 256)
        x2 = self.relu(self.conv_2_3(x2))  # (batch, 32, 256)
        x2 = self.maxpool_2_3(x2)  # (batch, 32, 128)
        x2 = self.relu(self.conv_2_4(x2))  # (batch, 32, 128)
        x2 = self.maxpool_2_4(x2)  # (batch, 32, 64)

        x = torch.cat([x1, x2], dim=1)  # (batch, 64, 64)
        x = self.dropout(x)
        return x  # (batch, 64, 64)


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
    - Dropout: 0.5 for regularization
    - Output Layer: Dense layer -> SBP and DBP (2 values)
    """
    def __init__(self):
        super().__init__()
        # CNN branches for ECG and PPG
        self.ecg_branch = CNNBranch()
        self.ppg_branch = CNNBranch()
        
        # LSTM layer: input 128 features, hidden size 64
        self.lstm = nn.LSTM(input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE, num_layers=1, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(OUTPUT_DROPOUT)
        
        # Output layer: dense layer for SBP and DBP
        self.output_layer = nn.Linear(LSTM_HIDDEN_SIZE, 2)
    
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
        ecg_features = self.ecg_branch(ecg)  # (batch, 64, 64)
        ppg_features = self.ppg_branch(ppg)  # (batch, 64, 64)
        
        # Stack features: concatenate along channel dimension
        combined = torch.cat([ecg_features, ppg_features], dim=1)  # (batch, 128, 64)
        
        # Reshape for LSTM: (batch, seq_len, features)
        combined = combined.permute(0, 2, 1)  # (batch, 64, 128)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # (batch, 128)
        
        # Dropout on last hidden state
        last_hidden = self.dropout(last_hidden)

        # Output layer
        output = self.output_layer(last_hidden)  # (batch, 2) - SBP and DBP
        
        return output
