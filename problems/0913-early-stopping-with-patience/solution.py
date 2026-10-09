from math import inf

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = inf if mode == "min" else -inf
        self.bad_steps = 0

    def step(self, metric: float) -> bool:
        improved = (
            metric < self.best if self.mode == "min"
            else metric > self.best
        )

        if improved:
            self.best = metric
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        return self.bad_steps >= self.patience