import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    l = logits - torch.logsumexp(logits, dim = 1, keepdim = True)

    entropy_loss = -1*torch.mean(l.gather(1, targets.unsqueeze(1)).squeeze(1))

    return entropy_loss