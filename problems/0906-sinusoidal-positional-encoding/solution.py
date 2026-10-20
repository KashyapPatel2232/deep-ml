import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    position = torch.arange(0., seq_len, 1).unsqueeze(-1)
    div_term = torch.exp((torch.arange(0,d_model, 2) * (-math.log(10000)/d_model))).unsqueeze(-1)
    arg = position @ div_term.T
    output = torch.zeros(seq_len, d_model)
    for col in range(0, d_model, 2):
        output[:, col] = torch.sin(arg[:, col//2])
        output[:, col+1] = torch.cos(arg[:, col//2])
    return output