import numpy as np
def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	"""
	Perform RMSProp optimization update.
	
	Args:
		params: List of parameter values
		grads: List of gradients for each parameter
		cache: List of cache values (moving average of squared gradients)
		lr: Learning rate
		beta: Decay rate for moving average
		epsilon: Small constant for numerical stability
	
	Returns:
		Tuple of (updated_params, updated_cache)
	"""
	cache = np.array(cache, dtype = float)
	grads = np.array(grads, dtype = float)
	params = np.array(params, dtype = float)
	# Your code here
	cache = beta * cache + (1-beta)*(grads**2)
	lr_eff = (lr/(np.sqrt(cache) + epsilon))
	params = params - lr_eff * grads
	return params.tolist(), cache.tolist()