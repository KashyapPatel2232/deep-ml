import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	dk = Q.shape[-1]
	attn = Q @ K.T / np.sqrt(dk)
	if mask is not None:
		attn = np.where(mask == 0, -1e9, attn)

	stable_attn = attn - np.max(attn, axis = -1, keepdims = True)
	sft_attn = np.exp(stable_attn) / np.sum(np.exp(stable_attn), axis = -1, keepdims = True)
	out = sft_attn @ V 
	return out, sft_attn