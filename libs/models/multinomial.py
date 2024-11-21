import os
import sys

# We need he project root to import the functions
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from libs.models.logistic_regression import LogisticRegression
import numpy as np
from libs.math import softmax


class SoftmaxClassifier(LogisticRegression):
    def __init__(self, num_features :int, num_classes:int):
        self.parameters = np.random.normal(0,1e-3,(num_features, num_classes))

    def predict(self, X : np.array) -> np.array:
        """
        Function to compute the raw scores for each sample and each class.

        Args:
            X: it's the input data matrix. The shape is (N, H)

        Returns:
            scores: it's the matrix containing raw scores for each sample and each class. The shape is (N, K)
        """
        scores = X.dot(self.parameters) 
        return scores

    
    def predict_labels(self, X: np.array) -> np.array:
        """
        Function to compute the predicted class for each sample.
        
        Args:
            X: it's the input data matrix. The shape is (N, H)
            
        Returns:
            preds: it's the predicted class for each sample. The shape is (N,)
        """
        scores = self.predict(X) # Get raw scores
        softmax_scores = softmax(scores)# Apply softmax to get probabilities
        preds = np.argmax(softmax_scores, axis=1) # Get class with highest probability for each sample
        return preds
    
    @staticmethod
    def likelihood(preds: np.array, y_onehot: np.array) -> float:
        """
        Function to compute the cross entropy loss from the predicted labels and the true labels.

        Args:
            preds: it's the matrix containing probability for each sample and each class. The shape is (N, K)
            y_onehot: it's the label array in encoded as one hot vector. The shape is (N, K)

        Returns:
            loss: The scalar that is the mean error for each sample.
        """
        log_preds = np.log(preds + 1e-10)  # We added epsilon to avoid log(0)
        loss = -np.sum(y_onehot * log_preds)
        return loss

    
    def update_theta(self, gradient:np.array, lr:float=0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the jacobian of the cross entropy loss.
            lr: the learning rate.

        Returns:
            None
        """
        self.parameters -= lr * gradient  #We update weights by subtracting the gradient scaled by learning rate
        pass
    
    @staticmethod
    def compute_gradient(x: np.array, y : np.array, preds: np.array) -> np.array:
        """
        Function to compute gradient of the cross entropy loss with respect the parameters. 

        Args:
            x: it's the input data matrix. The shape is (N, H)
            y: it's the label array in encoded as one hot vector. The shape is (N, K)
            preds: it's the predicted labels. The shape is (N, K)

        Returns:
            jacobian: A matrix with the partial derivatives of the loss. The shape is (H, K)
        """
        jacobian = x.T.dot(preds - y)
        return jacobian
    
    