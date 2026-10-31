import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	P = np.asarray(P, dtype = float)
	Q = np.asarray(Q, dtype = float)
	posititve_P = P > 0
	entropy = -np.sum(P[posititve_P] * np.log(P[posititve_P]))
	cross_entropy = -np.sum(P[posititve_P] * np.log(Q[posititve_P]))
	return entropy, cross_entropy