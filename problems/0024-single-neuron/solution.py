import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features = torch.tensor(features).float()
    weights = torch.tensor(weights)
    bias = torch.tensor(bias)
    labels = torch.tensor(labels)

    l1 = features @ torch.transpose(weights, dim0 = -1, dim1 = 0) + bias
    activation = torch.sigmoid(l1)
    mse_loss = F.mse_loss(activation, labels)
    return activation.tolist(), mse_loss.item()