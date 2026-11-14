import torch

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    if cache is not None:
        K_cache, V_cache = cache
        K_new = torch.concatenate([K_cache, (x_new @ W_K).unsqueeze(0)], dim = 0)
        V_new = torch.concatenate([V_cache, (x_new @ W_V).unsqueeze(0)], dim = 0)
    else:
        K_new, V_new = (x_new @ W_K).unsqueeze(0), (x_new @ W_V).unsqueeze(0)
    q = x_new @ W_Q

    scores = q @ K_new.transpose(-2, -1) / (q.shape[-1]**0.5)
    attn = torch.softmax(scores, dim = -1)
    output = attn @ V_new
    cache = K_new, V_new
    return output, cache