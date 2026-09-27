import torch

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    sft = torch.exp(x)/sum(torch.exp(x))
    res = sft.tolist()
    mat = [[0. for _ in range(len(res))] for _ in range(len(res))]
    for j in range(len(mat)):
        for i in range(len(mat[0])):
            if j==i:
                mat[i][j] = res[i]*(1-res[i])
            else: 
                mat[i][j] = -res[i]*res[j]

    out = torch.tensor(mat)
    return out
