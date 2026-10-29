import torch

def focal_loss(logits, targets, gamma=2.0):
    # TODO: mean focal loss over the batch
    logits = torch.tensor(logits)
    targets = torch.tensor(targets)
    pt = torch.nn.functional.softmax(logits, dim=-1)
    out = torch.tensor([pt[i, targets[i]] for i in range(len(targets))])
    loss = ((torch.tensor(1.0) - out)**gamma) * torch.mean(-1*torch.log(out))
    return loss
