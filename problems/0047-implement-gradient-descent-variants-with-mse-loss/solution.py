import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m = len(y)
    # Your code here
    for i in range(n_epochs):

        if method == 'batch':
            pred = X @ weights
            error = pred - y

            gradient = (2/m)* X.T @ error
            weights = weights - learning_rate * gradient

        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X[start: end]
                y_batch = y[start:end]

                pred = X_batch @ weights
                error = pred - y_batch

                gradient = (2/len(y_batch)) * X_batch.T @ error
                weights = weights - learning_rate*gradient

        else:
            for i in range(m):
                pred = X[i] @ weights
                error = pred - y[i]
    
                gradient = 2* error * X[i]
                weights = weights - learning_rate * gradient

    return weights