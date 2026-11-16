import numpy as np

def sliding_kv_cache_update(cache_k, cache_v, current_len, new_k, new_v, window_size):
    """
    Args:
        cache_k, cache_v: shape (batch, num_heads, window_size, head_dim)
        current_len: int, valid length of current cache
        new_k, new_v: shape (batch, num_heads, new_len, head_dim)
        window_size: int, maximum cache length.
    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """

    cache_k = np.array(cache_k)
    cache_v = np.array(cache_v)
    new_k = np.array(new_k)
    new_v = np.array(new_v)
    
    batch, num_heads, _, head_dim = cache_k.shape
    new_len = new_k.shape[-2]
    
    updated_len = min(current_len + new_len, window_size)
    
    new_cache_k = np.zeros_like(cache_k)
    new_cache_v = np.zeros_like(cache_v)
    
    combined_k = np.concatenate([cache_k[:, :, :current_len, :], new_k], axis=-2)
    combined_v = np.concatenate([cache_v[:, :, :current_len, :], new_v], axis=-2)
    
    if combined_k.shape[-2] > window_size:
        combined_k = combined_k[:, :, -window_size:, :]
        combined_v = combined_v[:, :, -window_size:, :]
        
    new_cache_k[:, :, :updated_len, :] = combined_k
    new_cache_v[:, :, :updated_len, :] = combined_v
    
    mask = []
    for i in range(new_len):
        query_pos = max(updated_len - new_len + i, 0)
        row = [1 if j <= query_pos else 0 for j in range(updated_len)]
        mask.append(row)
        
    return {
        'cache_k': new_cache_k.tolist(),
        'cache_v': new_cache_v.tolist(),
        'current_len': int(updated_len),
        'mask': mask
    }