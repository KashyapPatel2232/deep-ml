import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling.

    Args:
        x: list of float values to quantize
        block_size: number of elements per scaling block

    Returns:
        dict with keys:
            'quantized': list of dequantized values (rounded to 4 decimals)
            'scales': list of per-block scale factors (rounded to 4 decimals)
    """
    length = len(x)
    output = []
    scales = []
    if length % block_size == 0:
        blocks = length // block_size
    else:
        x.extend([0.0 for i in range(block_size - length % block_size)])
        blocks = length // block_size + 1
    
    for i in range(blocks):
        temp = np.array(x[i * block_size : (i + 1) * block_size], dtype=float)
        scale = np.max(np.abs(temp)) / 6
        if scale != 0:
            scale = 2 ** np.ceil((np.log2(scale)))
        else:
            scale = 1.0
        
        scales.append(round(float(scale), 4))
        fp4_pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        scaled = temp / scale
        idx = np.argmin(np.abs(np.abs(scaled)[:, None] - fp4_pos), axis=1)
        dequant = (np.sign(scaled) * fp4_pos[idx] * scale )
        output.extend(np.round(dequant, 4).tolist())

    return {'quantized': output[:length], 'scales': scales}