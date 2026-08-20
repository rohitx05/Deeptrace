"""
MesoNet-4 Architecture (Afchar et al., 2018)
"MesoNet: a Compact Facial Video Forgery Detection Network"
Standard deepfake forensic baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Meso4(nn.Module):
    """Meso-4 deepfake detection model."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        # Layer 1
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Fully connected
        self.dropout1 = nn.Dropout(0.5)
        # Input 160x160 -> pool1: 80x80 -> pool2: 40x40 -> pool3: 20x20 -> pool4: 5x5 -> 16*5*5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 16)
        self.leaky = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 160, 160)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.leaky(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits.squeeze(-1)
