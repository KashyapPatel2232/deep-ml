from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
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