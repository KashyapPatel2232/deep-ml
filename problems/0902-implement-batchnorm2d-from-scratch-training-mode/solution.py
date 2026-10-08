def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d

    x = torch.tensor(x)
    gamma = torch.tensor(gamma)
    beta = torch.tensor(beta)

    mean = x.mean(dim = (0,2,3), keepdims = True)
    var = x.var(dim = (0,2,3), keepdims = True, unbiased = False)

    norm = (x - mean) / torch.sqrt(var + eps)

    gamma = gamma.reshape(1,-1,1,1)
    beta = beta.reshape(1,-1,1,1)

    norm_x = norm * gamma + beta

    return norm_x