import numpy as np

def fit(model, x: np.array, y: np.array, x_val: np.array = None, y_val: np.array = None, lr: float = 0.5, num_steps: int = 500):
    """
    Function to fit the logistic regression model using gradient ascent.

    Args:
        model: the logistic regression model.
        x: input data matrix.
        y: label array.
        x_val: input data matrix for validation.
        y_val: label array for validation.
        lr: learning rate.
        num_steps: number of iterations.

    Returns:
        likelihood_history: values of the log likelihood during training.
        val_loss_history: values of validation loss during training (if validation data is provided).
    """
    likelihood_history = np.zeros(num_steps)
    val_loss_history = np.zeros(num_steps)

    for it in range(num_steps):
        # Make predictions using the model
        preds = model.predict(x)

        # Compute the gradient of the log-likelihood
        gradients = model.compute_gradient(x, y, preds)

        # Update the model's parameters (weights)
        model.update_theta(gradients, lr)

        # Compute and record the log-likelihood
        likelihood_history[it] = model.likelihood(preds, y)

        # If validation data is provided, compute and record validation loss
        if x_val is not None and y_val is not None:
            val_preds = model.predict(x_val)
            val_loss_history[it] = -model.likelihood(val_preds, y_val)

    return likelihood_history, val_loss_history

