import math

def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    # TODO: piecewise linear warmup, then half-cosine decay
    if step > total_steps:
        return 0.0
    
    if step >= 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    else:
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps)/ (total_steps - warmup_steps)))