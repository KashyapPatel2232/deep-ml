import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    # Your code here
    N, C, H, W = x.shape
    N, C, H, W = x.shape

    out_height = (H - kernel_size + stride - 1) // stride + 1
    out_width = (W - kernel_size + stride - 1) // stride + 1
    output = np.empty((N, C, out_height, out_width), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for out_y in range(out_height):
                for out_x in range(out_width):
                    start_y = out_y * stride
                    start_x = out_x * stride
                    end_y = min(start_y + kernel_size, H)
                    end_x = min(start_x + kernel_size, W)

                    window = x[n, c, start_y:end_y, start_x:end_x]
                    output[n, c, out_y, out_x] = np.max(window)

    return output