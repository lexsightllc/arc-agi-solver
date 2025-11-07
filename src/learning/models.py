import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class GridEncoder(nn.Module):
    """Encodes an ARCGrid into a feature vector."""
    def __init__(self, input_channels: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1))
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, input_channels, H, W)"""
        for i, layer in enumerate(self.conv_layers):
            x = F.relu(layer(x))
            if i % 2 == 1: # After BatchNorm
                x = F.max_pool2d(x, kernel_size=2, stride=2) # Downsample
        return x # (batch_size, hidden_dim, H', W')

class PolicyNet(nn.Module):
    """Predicts a distribution over DSL primitives and their arguments."""
    def __init__(self, input_channels: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.1, architecture: str = "resnet_transformer"):
        super().__init__()
        self.architecture = architecture
        self.grid_encoder = GridEncoder(input_channels, hidden_dim, num_layers)

        if architecture == "cnn":
            self.fc_policy = nn.Linear(hidden_dim * (30//(2**(num_layers//2)))**2, output_dim) # Assuming max 30x30 grid, downsampled
            self.fc_args = nn.Linear(hidden_dim * (30//(2**(num_layers//2)))**2, output_dim) # Placeholder for argument prediction
        elif architecture == "resnet_transformer":
            # Example: A simple transformer encoder on flattened features
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
            self.fc_policy = nn.Linear(hidden_dim, output_dim)
            self.fc_args = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, grid_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """grid_tensor: (batch_size, input_channels, H, W)
        Returns: (primitive_logits, arg_logits) - distributions over primitives and their arguments.
        """
        encoded_features = self.grid_encoder(grid_tensor)

        if self.architecture == "cnn":
            flat_features = encoded_features.view(encoded_features.size(0), -1)
            policy_logits = self.fc_policy(self.dropout(flat_features))
            arg_logits = self.fc_args(self.dropout(flat_features))
        elif self.architecture == "resnet_transformer":
            # Flatten spatial dimensions and treat as sequence for transformer
            batch_size, channels, H_prime, W_prime = encoded_features.shape
            seq_features = encoded_features.view(batch_size, channels, H_prime * W_prime).permute(0, 2, 1) # (B, H'*W', C)
            transformer_output = self.transformer_encoder(seq_features)
            # Pool or take first token for policy/arg prediction
            pooled_features = transformer_output.mean(dim=1) # (B, C)
            policy_logits = self.fc_policy(self.dropout(pooled_features))
            arg_logits = self.fc_args(self.dropout(pooled_features))

        return policy_logits, arg_logits

class ValueNet(nn.Module):
    """Predicts the expected value (e.g., probability of solving) of a given grid state."""
    def __init__(self, input_channels: int, hidden_dim: int, num_layers: int, output_dim: int, architecture: str = "cnn"):
        super().__init__()
        self.grid_encoder = GridEncoder(input_channels, hidden_dim, num_layers)
        if architecture == "cnn":
            self.fc_value = nn.Linear(hidden_dim * (30//(2**(num_layers//2)))**2, output_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """grid_tensor: (batch_size, input_channels, H, W)
        Returns: (batch_size, 1) - predicted value.
        """
        encoded_features = self.grid_encoder(grid_tensor)
        flat_features = encoded_features.view(encoded_features.size(0), -1)
        value = self.fc_value(flat_features)
        return value

class ProgramPriorModel(nn.Module):
    """A small neural network to learn a prior over programs or program fragments.
    This could take features from the policy network and predict a scalar likelihood.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, program_embedding: torch.Tensor) -> torch.Tensor:
        """program_embedding: (batch_size, input_dim) - e.g., from policy network's pooled features.
        Returns: (batch_size, 1) - logit for program likelihood.
        """
        x = F.relu(self.fc1(program_embedding))
        return self.fc2(x)
