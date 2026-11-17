import torch

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Apply a combined sampling pipeline to raw logits using PyTorch.
    
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
    # Your code here
    z = torch.tensor(logits, dtype = torch.float64)

    if temperature == 0:
        idx = int(torch.argmax(z))
        probs = torch.zeros_like(z)
        probs[idx] = 1.0
        return {"probabilities": [round(p, 4) for p in probs.tolist()], "sampled_token": idx}

    z = z/temperature

    if 0 < top_k < z.numel():
        vals, idx = torch.topk(z, top_k)
        z = torch.full_like(z, float("-inf")).scatter_(0, idx, vals)

    probs = torch.softmax(z, dim = -1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending = True)
        mask = (torch.cumsum(sorted_probs, dim = -1) - sorted_probs) >= top_p
        mask[0] = False
        probs = probs.scatter(0, sorted_idx, sorted_probs.masked_fill(mask, 0.0))
        probs = probs / probs.sum()

    gen = torch.Generator(device = z.device).manual_seed(seed)
    token = int(torch.multinomial(probs, 1, generator = gen))

    return {"probabilities": [round(p, 4) for p in probs.tolist()], "sampled_token": token}