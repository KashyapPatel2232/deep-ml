import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    anchor = torch.tensor(anchor)
    positive = torch.tensor(positive)
    negative = torch.tensor(negative)
    p1 = torch.sum((anchor - positive)**2)
    n1 = torch.sum((anchor - negative)**2)
    loss = torch.clamp(p1 - n1 + torch.tensor(margin), min = 0.0)
    return loss