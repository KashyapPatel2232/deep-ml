import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    # TODO
    layers = nn.Sequential(
        nn.Linear(2,2),
        nn.ReLU(),
        nn.Linear(2,1)
    )
    with torch.no_grad():
        layers[0].weight = nn.Parameter(w1)
        layers[0].bias = nn.Parameter(b1)
        layers[2].weight = nn.Parameter(w2)
        layers[2].bias = nn.Parameter(b2)

    output = layers(x)
    return output.item()