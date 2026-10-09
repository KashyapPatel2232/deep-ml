import torch

class StepLRScheduler:
    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        # Initialize initial_lr, step_size, and gamma using PyTorch tensors
        self.step_size = torch.tensor(step_size)
        self.initial_lr = torch.tensor(initial_lr)
        self.gamma = torch.tensor(gamma)
        

    def get_lr(self, epoch: int) -> float:
        # Calculate and return the learning rate for the given epoch
        return round((self.initial_lr * (self.gamma ** int(epoch / self.step_size))).float().item(), 4)