import os
import sys

# Trova il percorso della directory principale del progetto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features: int):
        self.parameters = np.random.normal(0, 0.01, num_features)

    def predict(self, x: np.array) -> np.array:
        """Computes predictions for input features."""
        thetaX = np.dot(x, self.parameters)
        preds = sigmoid(thetaX)
        return preds

    @staticmethod
    def likelihood(preds: np.array, y: np.array) -> np.array:
        """Computes the log-likelihood."""
        epsilon = 1e-10  # Avoid log(0)
        log_l = np.sum(y * np.log(preds + epsilon) + (1 - y) * np.log(1 - preds + epsilon))
        return log_l / (len(y) + epsilon)

    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """Updates model parameters using gradient ascent."""
        self.parameters += lr * gradient

    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """Computes the gradient of the log-likelihood."""
        errors = y - preds
        gradient = np.dot(x.T, errors) / len(y)
        return gradient