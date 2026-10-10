import torch

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr: float, T_max: int, min_lr: float):
        # Initialize initial_lr, T_max, and min_lr
        self.intial_lr = torch.tensor(initial_lr)
        self.T_max = torch.tensor(T_max)
        self.min_lr = torch.tensor(min_lr)

    def get_lr(self, epoch: int) -> float:
        # Calculate and return the learning rate for the given epoch, rounded to 4 decimal places
        return round((self.min_lr + 0.5*(self.intial_lr - self.min_lr) * (1 + (torch.cos(torch.pi * epoch / self.T_max)))).item(), 4)