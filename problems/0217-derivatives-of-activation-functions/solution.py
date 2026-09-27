import math
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	alpha = x
	beta = x
	gamma = x
	s1 = math.exp(alpha)/(1+math.exp(alpha))*(1-math.exp(alpha)/(1+math.exp(alpha)))
	s2 = 1- (math.tanh(beta))**2
	s3 = 0
	if gamma <= 0:
		s3 = 0. 
	else:
		s3 = 1.
	return {'sigmoid':s1, 'tanh':s2, 'relu':s3}