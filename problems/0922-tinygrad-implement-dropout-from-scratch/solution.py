from tinygrad import Tensor

def dropout(x: Tensor, p: float, training: bool) -> Tensor:
    if not training or p == 0.0:
        return x
    
    mask = (Tensor.rand(*x.shape, requires_grad=False) >= p)
    return x * mask / (1.0 - p)