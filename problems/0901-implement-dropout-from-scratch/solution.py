import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    
    if training and p != 0:
        keep_prob = 1-p
        mask = torch.rand(x.shape, dtype = x.dtype) < keep_prob
        out = x * mask / keep_prob

    else:
        return x
    return out