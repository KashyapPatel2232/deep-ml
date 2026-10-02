import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    W = torch.tensor(W)
    x = torch.tensor(x)
    grad_output = torch.tensor(grad_output)
    grad_input = grad_output @ W
    grad_W = grad_output.T @ x
    grad_b = grad_output.sum(dim = 0)

    return grad_input, grad_W, grad_b
