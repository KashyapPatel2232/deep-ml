import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    X = np.asarray(X)
    W_query = np.asarray(W_query)
    W_key = np.asarray(W_key)
    W_value = np.asarray(W_value)

    # Your code here
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    attn = Q @ K.swapaxes(-1, -2)/ np.sqrt(Q.shape[-1])
    B, T, _ = attn.shape
    mask_indices = np.triu_indices(T, k = 1)
    attn[:, mask_indices[0], mask_indices[1]] = -1e9
    
    mask_stable = attn - np.max(attn, axis = -1, keepdims = True)
    out = np.exp(mask_stable) / np.sum(np.exp(mask_stable), axis=-1, keepdims=True)
    context = out @ V
    return context
