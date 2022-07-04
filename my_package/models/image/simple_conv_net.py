from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        fc1_shape: List[int] = [256, 120],
        fc2_shape: List[int] = [120, 84],
        fc3_shape: List[int] = [84, 10],
    ):
        super(SimpleConvNet, self).__init__()
        if any(len(list_) != 2 for list_ in [fc1_shape, fc2_shape, fc3_shape]):
            raise ValueError(f"Length of fc*_shape must be 2.")
        if fc1_shape[0] != 256:
            raise ValueError(
                "Length of fc1_shape[0] must be 256, assuming input size is 28x28."
            )
        if fc3_shape[-1] != 10:
            raise ValueError(
                "Length of fc3_shape[1] must be 10,"
                " assuming used for MNIST classification."
            )

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(*fc1_shape)
        self.fc2 = nn.Linear(*fc2_shape)
        self.fc3 = nn.Linear(*fc3_shape)

    def forward(self, x):
        # MNIST input size will be [bs, 1, 28, 28]
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
