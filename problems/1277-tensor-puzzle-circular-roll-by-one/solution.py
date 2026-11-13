import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    # Your code here
    n = a.shape[0]
    return a[(np.arange(n)+1)%n]
