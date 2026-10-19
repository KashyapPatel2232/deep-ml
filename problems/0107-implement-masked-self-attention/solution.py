import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	attention = Q @ K.T / (K.shape[-1] ** 0.5) + mask
	max_attention = attention - np.max(attention, axis = -1, keepdims = True)
	sft_attention = np.exp(max_attention) / np.sum(np.exp(max_attention), axis = -1, keepdims = True)
	output = sft_attention @ V
	return output