import torch.nn as nn
import torch

class CorrectedCNN(nn.Module):
    def __init__(self):
        """
        Initializes the CorrectedCNN model with convolutional, pooling, and fully connected layers.
        """
        super(CorrectedCNN, self).__init__()
        # First convolutional layer with fewer filters
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2)  # Fewer filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling

        # Second convolutional layer with fewer filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)  # Fewer filters

        # Fully connected layer with reduced dimensions
        self.fc1 = nn.Linear(16 * 2 * 2, 10)  # Smaller dimensions to reduce capacity

    def forward(self, x):
        """
        Defines the forward pass of the model: applies convolutional layers, pooling, 
        flattening, and then a fully connected layer to produce the output.
        """
        x = self.pool(self.conv1(x))  # First block
        x = self.pool(self.conv2(x))  # Second block
        x = x.view(-1, 16 * 2 * 2)  # Flatten
        x = self.fc1(x)  # Final output
        return x
