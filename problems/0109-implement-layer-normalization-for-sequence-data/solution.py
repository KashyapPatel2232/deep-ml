import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	mean = X.mean(axis = 2, keepdims = True)
	var = X.var(axis = 2, keepdims = True)
	norm = (X - mean)/np.sqrt(var + epsilon)
	norm_X = norm * gamma + beta
	return norm_X