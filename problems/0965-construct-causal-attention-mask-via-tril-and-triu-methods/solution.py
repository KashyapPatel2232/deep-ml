import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    if method == 'tril':
        M = np.zeros_like(attn_weights)
        for i in range(len(attn_weights)):
            for j in range(len(attn_weights[0])):
                if j <= i:
                    M[i][j] = 1
                else:
                    M[i][j] = 0
        A_bar = attn_weights * M
        out = A_bar / np.sum(A_bar, axis=-1, keepdims=True)
    else:
        masked_weights = np.log(attn_weights)
        for i in range(len(masked_weights)):
            for j in range(len(masked_weights[0])):
                if j > i:
                    masked_weights[i][j] = -1e9
        out = np.exp(masked_weights) / np.sum(np.exp(masked_weights), axis=-1, keepdims=True)
        
    return out.tolist()