import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X = np.asarray(X)
    attention = X @ X.T
    stable = attention - np.max(attention, axis = -1, keepdims = True)
    sft_max = np.exp(stable) / np.sum(np.exp(stable), axis = -1, keepdims = True)
    context_vec = sft_max @ X
    return context_vec
