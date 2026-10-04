import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    model.zero_grad()
    out = model(x)
    loss = ((out - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        model.weight -= lr*model.weight.grad
        model.bias -= lr* model.bias.grad

    return loss.item()