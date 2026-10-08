import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    X = torch.tensor(x)
    gamma = torch.tensor(gamma)
    beta = torch.tensor(beta)

    # Instead of hardcoding 2 we should ensure to take it as last dim i.e. -1
    mean = x.mean(dim = -1, keepdims = True)
    var = ((x - mean)**2).mean(dim = -1, keepdims = True)
    norm = (x - mean) / torch.sqrt(var + eps)

    norm_x = norm * gamma + beta

    return norm_x
