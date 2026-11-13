import torch

def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a torch.Tensor
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    z = logits_tensor - logits_tensor.max(dim=-1, keepdim=True)[0]
    probs = torch.exp(z) / torch.sum(torch.exp(z), dim=-1, keepdim=True)

    sorted_probs, sorted_indices = torch.sort(probs, descending = True, dim = -1)
    cum_probs = torch.cumsum(sorted_probs, dim =-1)

    keep_mask = (cum_probs - sorted_probs) < p
    keep_mask[..., 0] = True

    filtered_sorted_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
    renormalized_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(probs)
    output.scatter_(dim = -1, index = sorted_indices ,src = renormalized_sorted_probs)

    return output