import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    attention = Q @ np.swapaxes(K, -2, -1)/ (K.shape[-1] ** 0.5)
    norm_attn = attention - np.max(attention, axis = -1, keepdims = True)
    self_attention = np.exp(norm_attn)/ np.sum(np.exp(norm_attn), axis = -1, keepdims = True)
    output = self_attention @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    Q = np.swapaxes(Q.reshape(seq_len, n_heads, -1), 0, 1)
    K = np.swapaxes(K.reshape(seq_len, n_heads, -1), 0, 1)
    V = np.swapaxes(V.reshape(seq_len, n_heads, -1), 0, 1)

    output = self_attention(Q, K, V)

    return np.swapaxes(output, 0, 1).reshape(seq_len, d_model)
