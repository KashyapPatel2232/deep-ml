import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    if cache is not None:
        K_cache, V_cache = cache
        K_new = np.concatenate([K_cache, np.asarray([x_new @ W_K])], axis = 0)
        V_new = np.concatenate([V_cache, np.asarray([x_new @ W_V])], axis = 0)
    else:
        K_new, V_new = np.asarray([x_new @ W_K]), np.asarray([x_new @ W_V])
    q = x_new @ W_Q

    scores = q @ K_new.swapaxes(-2, -1) / (q.shape[-1]**0.5)
    scores_max = scores - np.max(scores, axis = -1, keepdims = True)
    attn = np.exp(scores_max) / np.sum(np.exp(scores_max), axis = -1, keepdims = True)
    output = attn @ V_new
    cache = K_new, V_new
    return output, cache