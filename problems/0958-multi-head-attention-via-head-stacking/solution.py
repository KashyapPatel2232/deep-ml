import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    multi_head_output = []
    # Your code here
    for i in range(len(heads)):
        W_q, W_k, W_v = np.asarray(heads[i])
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        attn = Q @ K.swapaxes(-2, -1) / np.sqrt(Q.shape[-1])
        B, T, _ = attn.shape
        mask_indices = np.triu_indices(T, k = 1)
        attn[:, mask_indices[0], mask_indices[1]]= -1e9
        stable_attn = attn - np.max(attn, axis = -1, keepdims = True)
        sft = np.exp(stable_attn) / np.sum(np.exp(stable_attn), axis = -1, keepdims = True)
        context = sft @ V
        multi_head_output.append(context)
    output = np.asarray(multi_head_output).swapaxes(0, 1).swapaxes(1, 2)
    return output.reshape(B, T, -1)
