import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	out1 = w1 @ x

	out2 = out1 * (out1 > 0)

	out3 = w2 @ out2 + x

	output = out3 * (out3 > 0)

	return output
