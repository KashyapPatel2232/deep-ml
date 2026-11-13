import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    gamma1 = params['ln1_gamma']
    gamma2 = params['ln2_gamma']
    beta1 = params['ln1_beta']
    beta2 = params['ln2_beta']

    ln1 = gamma1 * ((x-np.mean(x, axis = -1, keepdims = True))/np.sqrt(np.std(x, axis = -1, keepdims = True)**2 + 1e-5)) + beta1

    W_q = params['W_q']
    W_k = params['W_k']
    W_v = params['W_v']
    W_o = params['W_o']

    batch, seq_len, emb_dim = x.shape

    Q = (ln1 @ W_q).reshape(batch, seq_len, num_heads, emb_dim//num_heads).swapaxes(1, 2)
    K = (ln1 @ W_k).reshape(batch, seq_len, num_heads, emb_dim//num_heads).swapaxes(1, 2)
    V = (ln1 @ W_v).reshape(batch, seq_len, num_heads, emb_dim//num_heads).swapaxes(1, 2)

    attn = Q @ K.swapaxes(-1, -2) / np.sqrt(emb_dim//num_heads)
    attn_max = attn - np.max(attn, axis = -1, keepdims = True)
    sft_attn = np.exp(attn_max) / np.sum(np.exp(attn_max), axis = -1, keepdims = True)
    output = np.transpose((sft_attn @ V), axes = (0, 2, 1, 3)).reshape(batch, seq_len, emb_dim)

    MHA = output @ W_o

    z = x + MHA

    W_ff1 = params['W_ff1']
    b_ff1 = params['b_ff1']
    W_ff2 = params['W_ff2']
    b_ff2 = params['b_ff2']

    ln2 = gamma1 * ((z-np.mean(z, axis = -1, keepdims = True))/np.sqrt(np.std(z, axis = -1, keepdims = True)**2 + 1e-5)) + beta1

    l1 = ln2 @ W_ff1 + b_ff1
    gelu = 0.5 * l1 * (1+ np.tanh((2/np.pi)**0.5 * (l1 + 0.044715 * (l1**3)))) 
    h = gelu @ W_ff2 + b_ff2

    output = z + h

    return output