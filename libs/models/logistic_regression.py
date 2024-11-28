import os
import sys

# This code is needed to access the correct path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features: int):
        self.parameters = np.random.normal(0, 0.01, num_features)

    def predict(self, x: np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        thetaX = np.dot(x, self.parameters)
        preds = sigmoid(thetaX)
        return preds

    @staticmethod
    def likelihood(preds: np.array, y: np.array) -> np.array:
        """
         Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        epsilon = 1e-10  # Avoid log(0)
        log_l = np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) 
        return log_l   

    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        self.parameters += lr * gradient

    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        errors = y - preds
        gradient = np.dot(x.T, errors) / len(y)
        return gradient