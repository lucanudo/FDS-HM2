import torch.nn as nn
import torch

class CorrectedCNN(nn.Module):
    def __init__(self):
        super(CorrectedCNN, self).__init__()
        # Primo livello convolutivo con meno filtri
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2)  # Meno filtri
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling

        # Secondo livello convolutivo con meno filtri
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)  # Meno filtri

        # Livello completamente connesso ridotto
        self.fc1 = nn.Linear(16 * 2 * 2, 10)  # Dimensioni più piccole per ridurre la capacità

    def forward(self, x):
        x = self.pool(self.conv1(x))  # Primo blocco
        x = self.pool(self.conv2(x))  # Secondo blocco
        x = x.view(-1, 16 * 2 * 2)  # Flatten
        x = self.fc1(x)  # Output finale
        return x
