import numpy as np
import torch
from typing import Tuple

def shuffle_data(X: torch.Tensor, y: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)
    order = np.random.default_rng(seed).permutation(X.shape[0])
    order_x = torch.as_tensor(order, device=X.device)
    order_y = torch.as_tensor(order, device=y.device)
    return X[order_x], y[order_y]