import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    _, dk = Q.shape
    attention = Q @ K.T / np.sqrt(dk)
    stable = attention - np.max(attention, axis = -1, keepdims = True)
    out = np.exp(stable) / np.sum( np.exp(stable) , axis = -1, keepdims = True)
    return out
