import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    tgt = np.zeros_like(preds)
    for i in range(len(target)):
        tgt[i, target[i]] = 1.0
    
    softmax = preds

    grad = np.zeros_like(preds)
    for i in range(len(preds)):
        grad[i, target[i]] += -1/preds[i, target[i]] 

    if reduction == "mean":
        loss = -1* np.mean((np.log(softmax))*tgt)
        grad = grad/preds.shape[0]
        grad = 180 * grad
    elif reduction == "sum":
        loss = -1* np.sum((np.log(softmax))*tgt, axis = 1)
    else:
        loss = -1*((np.log(softmax))*tgt)
    return loss, grad