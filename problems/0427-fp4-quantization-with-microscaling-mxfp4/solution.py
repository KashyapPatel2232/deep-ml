import torch

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    length = len(x)
    padded = list(x)
    if length % block_size != 0:
        padded.extend([0.0] * (block_size - length % block_size))

    blocks = len(padded) // block_size
    fp4_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)

    output = []
    scales = []

    for i in range(blocks):
        temp = torch.tensor(padded[i * block_size : (i + 1) * block_size], dtype=torch.float64)
        scale = torch.max(torch.abs(temp)) / 6.0

        if scale != 0:
            scale = 2.0 ** torch.ceil(torch.log2(scale))
        else:
            scale = torch.tensor(1.0, dtype=torch.float64)

        scales.append(round(float(scale), 4))
        scaled = temp / scale
        idx = torch.argmin(torch.abs(torch.abs(scaled)[:, None] - fp4_pos), dim=1)
        dequant = torch.sign(scaled) * fp4_pos[idx] * scale
        output.extend(torch.round(dequant, decimals=4).tolist())

    return {'quantized': output[:length], 'scales': scales}