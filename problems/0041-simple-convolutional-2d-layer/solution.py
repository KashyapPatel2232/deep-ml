import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    output = F.conv2d(input_matrix, kernel, stride = stride, padding=padding)
    return output.squeeze(0).squeeze(0)