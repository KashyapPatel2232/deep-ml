import math

def sigmoid(z: float) -> float:
	result = math.exp(z)/(1+math.exp(z))
	return result