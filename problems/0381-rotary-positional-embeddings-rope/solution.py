import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, d = x.shape
    index = np.arange(d//2, dtype = int)
    output = np.zeros_like(x)
    for row, pos in enumerate(positions):
        for i in index:
            theta = pos * (1 / (base ** (2*i / d)))
            output[row][2*i] = x[row][2*i] * np.cos(theta) - x[row][2*i + 1] * np.sin(theta)
            output[row][2*i + 1] = x[row][2*i] * np.sin(theta) + x[row][2*i + 1] * np.cos(theta)
    return output