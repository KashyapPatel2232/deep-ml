import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Apply a combined sampling pipeline to raw logits.
    
    Args:
        logits: Raw unnormalized scores for each token
        temperature: Scaling factor for logit sharpness (0 = greedy)
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dict with 'probabilities' (list of floats rounded to 4 decimals)
        and 'sampled_token' (int index of chosen token)
    """
    # For sampling, use: rng = np.random.default_rng(seed)
    # Your code here
    rng = np.random.default_rng(seed)
    z = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        best_idx = int(np.argmax(z))
        probs = np.zeros_like(z)
        probs[best_idx] = 1.0
        return {
            "probabilities": [round(float(p), 4) for p in probs],
            "sampled_token": best_idx,
        }

    z = z / temperature
    
    if len(z) > top_k > 0:
        top_k_indices = np.argpartition(z, -top_k)[-top_k:]
        filtered = np.full_like(z, -np.inf)
        filtered[top_k_indices] = z[top_k_indices]
        z = filtered

    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    probs = exp_z / np.sum(exp_z)

    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        remove_mask = (cumsum_probs - sorted_probs) >= top_p

        remove_mask[0] = False

        probs[sorted_indices[remove_mask]] = 0.0
        probs = probs / np.sum(probs)

    sample = rng.choice(len(probs), p = probs)
    return {
        "probabilities": [round(float(p), 4) for p in probs],
        "sampled_token": sample,
    }


    