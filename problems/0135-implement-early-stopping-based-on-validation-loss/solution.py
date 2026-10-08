import torch
from typing import Tuple

def early_stopping(val_losses: torch.Tensor, patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses.
    
    Args:
        val_losses: A 1D tensor of validation losses for each epoch
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum decrease in loss to qualify as an improvement
    
    Returns:
        Tuple of (stop_epoch, best_epoch)
    """
    best_loss = val_losses[0]
    best_epoch = 0
    stop_epoch = 0
    patience_count = 0
    for i in range(1, len(val_losses)):
        current_loss = val_losses[i]
        stop_epoch = i

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_epoch = i
            patience_count = 0

        else:
            patience_count += 1

        if patience_count == patience:
            break

    return stop_epoch, best_epoch