import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	accuracy = np.mean(y_true == y_pred)
	return accuracy