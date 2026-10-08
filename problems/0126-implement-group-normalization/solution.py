import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    B, C, H, W = X.shape
    grp_sz = C // num_groups
    X_reshape = X.reshape(B, num_groups, grp_sz, H, W)

    mean = X_reshape.mean(axis = (2,3,4), keepdims = True)
    var = X_reshape.var(axis = (2,3,4), keepdims = True)

    norm = (X_reshape - mean) / np.sqrt(var + epsilon)

    norm = norm.reshape(B, C, H, W)

    norm_x = norm * gamma + beta

    return norm_x
    