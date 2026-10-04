import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    buffer = io.BytesIO()
    torch.save(src.state_dict(), buffer)
    buffer.seek(0)
    loaded_state_dict = torch.load(buffer, weights_only = True)
    dst.load_state_dict(loaded_state_dict)
    return dst
