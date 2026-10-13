import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        ctx.save_for_backward(x)
        out = torch.maximum(torch.tensor(0.), x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        x,  = ctx.saved_tensors 
        return grad_output * (x > torch.tensor(0.))