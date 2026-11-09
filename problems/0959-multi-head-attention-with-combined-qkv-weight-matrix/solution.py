import numpy as np


def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Array of shape (batch, num_tokens, d_out).
    """

    batch_size, num_tokens, _ = x.shape
    d_out, output_width = W_out.shape


    head_dim = d_out // num_heads
    qkv = (x @ W_qkv).reshape(batch_size, num_tokens, 3, num_heads, head_dim)
    qkv = qkv.transpose(2, 0, 3, 1, 4)
    query, key, value = qkv

    scores = query @ key.swapaxes(-2, -1) / np.sqrt(head_dim)
    causal_mask = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)

    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ value
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, num_tokens, d_out)
    return context @ W_out