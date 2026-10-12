def conv_out_shape(h, w, kernel, stride, padding):
    """Return (H_out, W_out) for a 2D conv with the given spatial params.

    Args:
        h: input height
        w: input width
        kernel: kernel size (same for H and W)
        stride: stride (same for H and W)
        padding: padding (same for H and W)

    Returns:
        Tuple of ints (H_out, W_out).
    """
    # TODO: apply the standard conv output-size formula
    H_out = (h + 2* padding - kernel)//stride + 1
    W_out = (w + 2* padding - kernel)//stride + 1
    return H_out, W_out
