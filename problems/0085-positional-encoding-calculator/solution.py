import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    if position == 0 or d_model <= 0:
        return -1
    
    even = torch.arange(0, d_model, 2)
    output = []

    for pos in range(position):
        temp1 = torch.sin(pos / (10000.0 ** (even / d_model)))
        temp2 = torch.cos(pos / (10000.0 ** (even / d_model)))
        temp = torch.zeros(d_model)
        for i in range(d_model):
            if i % 2 == 0:
                temp[i] = temp1[i//2]
            else:
                temp[i] = temp2[i//2] 
        output.append(temp)
        
    return torch.stack(output).to(torch.float16)