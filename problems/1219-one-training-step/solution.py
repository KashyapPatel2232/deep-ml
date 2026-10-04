import torch

def train_step(model, x, y, optimizer, loss_fn):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # TODO: zero_grad -> forward -> loss -> backward -> step; return float loss
    model.zero_grad()
    out = model(x)
    loss = torch.mean(loss_fn(out,y))
    loss.backward()
    with torch.no_grad():
        optimizer.step()
    return loss.item()
