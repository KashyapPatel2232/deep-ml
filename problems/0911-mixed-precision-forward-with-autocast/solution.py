import torch

def forward_bf16(model, x):
    # TODO: run model(x) inside torch.amp.autocast with dtype=bfloat16; return float32 output
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(x)
    return out.float()
