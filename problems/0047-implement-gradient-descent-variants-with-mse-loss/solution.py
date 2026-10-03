import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """
    # Your implementation here
    m = len(y)

    for i in range(n_epochs):

        if method == 'batch':
            preds = X @ weights
            error = preds - y

            gradient = X.T @ error
            weights = weights - (2/m)*learning_rate*gradient

        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]

                preds = X_batch @ weights
                error = preds - y_batch

                gradient = X_batch.T @ error
                weights = weights - (2/batch_size) * learning_rate * gradient

        else:
            for k in range(m):
                preds = X[k] @ weights
                error = preds - y[k]

                gradient = 2 * X[k] * error
                weights = weights - learning_rate*gradient

    return weights

