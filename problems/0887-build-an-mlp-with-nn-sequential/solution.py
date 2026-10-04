import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # TODO: return a Sequential of Linear -> ReLU -> Linear
    out = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(hidden_dim),
        nn.Linear(hidden_dim, out_dim)
    )
    return out