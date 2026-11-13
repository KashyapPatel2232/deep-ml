import torch
import torch.nn.functional as F

def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1
        return probs

    return F.softmax(logits / temperature, dim=-1)