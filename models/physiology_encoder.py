"""
Physiological Signal Encoder.
Checks biological consistency via green-channel PPG signal analysis.
Only used for video (temporal) inputs.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PhysiologyEncoder(nn.Module):
    """
    Lightweight PPG-based physiological consistency checker.
    Analyzes green-channel temporal variations to detect unnatural
    patterns that indicate deepfake manipulation.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim

        # LSTM to process green-channel temporal signal
        self.signal_rnn = nn.LSTM(
            input_size=1,  # green channel mean per frame
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Projection to feature space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
        )

    def extract_ppg_signal(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract PPG-like signal from green channel variance across frames.

        Args:
            frames: (B, T, 3, H, W) video frames

        Returns:
            signal: (B, T, 1) green-channel mean per frame
        """
        # Extract green channel mean for each frame
        green = frames[:, :, 1, :, :]  # (B, T, H, W)
        signal = green.mean(dim=(-2, -1), keepdim=False)  # (B, T)
        return signal.unsqueeze(-1)  # (B, T, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) video frames (normalized)

        Returns:
            features: (B, feature_dim) physiological consistency features
        """
        signal = self.extract_ppg_signal(frames)  # (B, T, 1)
        rnn_out, _ = self.signal_rnn(signal)  # (B, T, hidden*2)
        # Take last timestep
        last_hidden = rnn_out[:, -1, :]  # (B, hidden*2)
        features = self.projection(last_hidden)  # (B, feature_dim)
        return features
