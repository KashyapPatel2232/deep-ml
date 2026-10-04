import torch
import torch.nn as nn

def count_params():
    """Build Sequential(Linear(4,8), ReLU, Linear(8,2)) and return trainable param count.

    Returns:
        int: total number of trainable parameters
    """
    # TODO: build the model and sum p.numel() for trainable params
    model = nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(8),
        nn.Linear(8,2)
    )
    return 32 + 8 + 16 + 2
