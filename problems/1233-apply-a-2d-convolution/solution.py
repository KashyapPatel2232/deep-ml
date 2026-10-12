import torch
import torch.nn as nn


def apply_conv2d():
    """Build nn.Conv2d(1, 1, kernel_size=2, bias=False), set a fixed kernel, convolve a fixed input.

    Under torch.no_grad(), set weight to tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).
    Input is tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]).

    Returns:
        torch.Tensor: Output of shape (1, 1, 2, 2).
    """
    # TODO: build conv, set weight under no_grad, apply to fixed input
    layer = nn.Conv2d(1,1, kernel_size = 2, bias = False)
    with torch.no_grad():
        layer.weight.data = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    return layer(torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])).detach()