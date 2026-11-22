import numpy as np

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack even-length nibble array (0..15) into uint8 bytes: lo | (hi<<4)."""
    # Your code here
    output = []
    for i in range(0, len(q), 2):
        lo = q[i]
        hi = q[i+1]
        out = lo | (hi << 4)
        output.append(out)
    return np.array(output, dtype = np.uint8)

def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack bytes to length-n int nibbles."""
    output = []
    # Your code here
    for byte in packed:
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        output.append(lo)
        output.append(hi)
    return np.array(output[:n], dtype = int)

def w4a16_matmul(x, packed_w, scales, in_features: int):
    """
    x: (batch, in_features) float
    packed_w: (out_features, in_features//2) uint8
    scales: (out_features,)
    Dequant: W = scale * (q - 8), y = x @ W.T
    """
    # Your code here
    x = np.asarray(x, dtype = float)
    packed_w = np.asarray(packed_w, dtype = np.uint8)
    scales = np.asarray(scales, dtype = float)

    unpacked = np.array([unpack_int4(row, in_features) for row in packed_w], dtype = float)
    W = scales[:, None] * (unpacked - 8.0)
    output = x @ W.T
    return output
