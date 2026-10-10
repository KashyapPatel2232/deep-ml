import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    total_loss = 0
    for x_batch, targets in micro_batches:
        output = model(x_batch)
        loss = (1/len(micro_batches))*criterion(output, targets)
        total_loss += loss.item()
        loss.backward()

    with torch.no_grad():
        optimizer.step()

    return total_loss