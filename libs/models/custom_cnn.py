import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import datasets

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First convolutional layer with batch normalization and dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional layer with batch normalization and dropout
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional layer with batch normalization and dropout
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.25)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted to the correct input size after pooling
        self.fc2 = nn.Linear(512, 10)  # Output layer for 10 classes
    
    def forward(self, x):
        # Pass through the first convolutional layer, batch normalization, and dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Pass through the second convolutional layer, batch normalization, and dropout
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Pass through the third convolutional layer, batch normalization, and dropout
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layer 1
        x = F.relu(self.fc1(x))
        
        # Output layer (no activation, will use CrossEntropyLoss which applies softmax)
        x = self.fc2(x)
        
        return x