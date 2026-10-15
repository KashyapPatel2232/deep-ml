import numpy as np

def sequence_mask(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Zero out positions j >= lengths[i] in each row i."""
    # Your code here

    x_indices = np.indices(values.shape)[1]
    length = lengths.reshape((-1,1))
    output = (x_indices < length) * values
    return output