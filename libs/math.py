import numpy as np

def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """

    g = 1 / (1 + np.exp(-x))
    return g


def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted scores (logits), a 2D array of shape (N, K), where:
           - N is the number of samples
           - K is the number of classes

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """
    # Compute the exponential of the input scores, subtracting the max for numerical stability
    exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    
    # Normalize to obtain probabilities
    softmax_scores = exp_y / np.sum(exp_y, axis=1, keepdims=True)
    return softmax_scores
