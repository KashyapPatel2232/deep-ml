import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape

    num_heads = 4
    assert dim % num_heads == 0
    head_dim = dim // num_heads
    
    Q_split = Q.reshape(batch_size, query_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_split = K.reshape(batch_size, key_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_split = V.reshape(batch_size, key_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    

    scores = Q_split @ K_split.transpose(0, 1, 3, 2)
    
    scaled_scores = scores / np.sqrt(head_dim)
    
    max_scores = np.max(scaled_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scaled_scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    head_outputs = attention_weights @ V_split
    
    output = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_len, dim)
    
    return output
