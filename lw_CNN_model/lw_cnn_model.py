import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Inception Block (Time-Frequency Feature Learning)
# -------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.branch1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)
        )

        self.branch2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=1
        )

        self.batch_norm = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.batch_norm(out)
        return F.relu(out)


# -------------------------------------------------
# Final Lung Sound CNN
# -------------------------------------------------
class LungSoundCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Inception Blocks
        self.incep1 = InceptionBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.incep2 = InceptionBlock(96, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.incep3 = InceptionBlock(192, 128)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected
        self.fc1 = nn.Linear(384, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        # Input: (batch, 3, 128, 216)
        # CNN expects (batch_size, channels, height, width)

        x = F.relu(self.bn1(self.conv1(x)))  # (B,32,128,216)

        x = self.pool1(self.incep1(x))  # (B,96,64,108)
        x = self.pool2(self.incep2(x))  # (B,192,32,54)
        x = self.incep3(x)  # (B,384,32,54)

        x = self.global_pool(x)  # (B,384,1,1)
        x = torch.flatten(x, 1)  # (B,384)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)  # (B,num_classes)

        return x
