import torch.nn as nn
import torch

class CorrectedCNN(nn.Module):
    def __init__(self):
        super(CorrectedCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input channels: 3 (RGB), Output channels: 16
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling reduces spatial dimensions by half

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input channels: 16, Output channels: 32
        self.relu2 = nn.ReLU()

        # Fully connected layer
        self.fc1 = nn.Linear(32 * 8 * 8, 10)  # 32 * 8 * 8 matches the output of the final convolutional layer, 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))  # Output: (16, 16, 16) after first convolution and pooling
        x = self.pool(self.relu2(self.conv2(x)))  # Output: (32, 8, 8) after second convolution and pooling
        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)  # Final output layer
        return x