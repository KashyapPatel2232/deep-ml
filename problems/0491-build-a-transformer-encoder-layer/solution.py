import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']
    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    gamma1 = weights['gamma1']
    gamma2 = weights['gamma2']
    beta1 = weights['beta1']
    beta2 = weights['beta2']

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    batch_size, seq_len, d_model = Q.shape

    Q = Q.reshape(batch_size, seq_len, num_heads, d_model//num_heads).swapaxes(1, 2)
    K = K.reshape(batch_size, seq_len, num_heads, d_model//num_heads).swapaxes(1, 2)
    V = V.reshape(batch_size, seq_len, num_heads, d_model//num_heads).swapaxes(1, 2)

    attn = Q @ K.swapaxes(-2, -1) / np.sqrt(d_model//num_heads)
    attn_max = attn - np.max(attn, axis = -1, keepdims = True)
    sft_attn = np.exp(attn_max) / np.sum(np.exp(attn_max), axis = -1, keepdims = True)
    output = np.transpose((sft_attn @ V), axes = (0, 2, 1, 3)).reshape(batch_size, seq_len, d_model)

    output = output @ W_o

    Z = ((X + output - np.mean(X + output, axis = -1, keepdims = True)) / np.sqrt(np.std(X + output, axis = -1, keepdims = True)**2+ eps) )* gamma1 + beta1
    
    FFN = Z + (np.maximum(Z @ W1 + b1, 0.0)) @ W2 + b2
    output = ((FFN - np.mean(FFN, axis = -1, keepdims = True)) / np.sqrt(np.std(FFN, axis = -1, keepdims = True)**2+ eps) )* gamma2 + beta2

    return output