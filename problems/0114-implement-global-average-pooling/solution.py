import numpy as np
def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	H, W, C = x.shape
	output = np.zeros(C)
	for i in range(C):
		output[i] = np.mean(x[:,:,i], axis = (0,1)).item()

	return output