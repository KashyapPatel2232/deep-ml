import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits = np.asarray(logits, dtype=np.float64)

    # 1. Numerically stable softmax
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    cum_probs = np.cumsum(sorted_probs, axis=-1)

    keep_mask = (cum_probs - sorted_probs) < p
    keep_mask[..., 0] = True
    filtered_sorted_probs = np.where(keep_mask, sorted_probs, 0.0)
    renormalized_sorted_probs = filtered_sorted_probs / np.sum(
        filtered_sorted_probs, axis=-1, keepdims=True
    )
    output = np.zeros_like(probs)
    np.put_along_axis(output, sorted_indices, renormalized_sorted_probs, axis=-1)

    return output.tolist()