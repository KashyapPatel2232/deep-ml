import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    out = torch.max(t, dim, keepdim=True)[0]
    res = t - out
    return torch.exp(res)/torch.sum(torch.exp(res), dim, keepdim=True)