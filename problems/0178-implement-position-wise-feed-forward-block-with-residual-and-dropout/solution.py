import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.

    Args:
        x: input vector
        W1, b1: first linear layer parameters
        W2, b2: second linear layer parameters
        dropout_p: dropout probability
        seed: random seed for reproducibility

    Returns:
        Output vector after FFN block (rounded to 4 decimals)
    """
    # Your code here
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)

    hidden = W1 @ x + b1
    relu = np.maximum(hidden, 0.0)
    out = W2 @ relu + b2

    if dropout_p > 0:
        np.random.seed(seed)
        mask = (np.random.rand(*out.shape) > dropout_p) / (1.0- dropout_p)
        out = out * mask
    
    output = out + x
    return output