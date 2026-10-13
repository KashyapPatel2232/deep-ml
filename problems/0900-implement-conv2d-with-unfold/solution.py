import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    batch_size, in_channels, height, width = x.shape
    out_channels, weight_in_channels, kernel_h, kernel_w = W.shape


    height_out = (height + 2 * padding - kernel_h) // stride + 1
    width_out = (width + 2 * padding - kernel_w) // stride + 1

    patches = F.unfold(
        x,
        kernel_size=(kernel_h, kernel_w),
        padding=padding,
        stride=stride,
    )
    weight_flat = W.reshape(out_channels, -1).unsqueeze(0)

    output = weight_flat @ patches
    output = output.reshape(batch_size, out_channels, height_out, width_out)

    if b is not None:
        output = output + b.reshape(1, out_channels, 1, 1)

    return output