import torch

def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    fn = torch.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0

    score = (1 + beta_squared) * precision * recall / denominator
    return round(score.item(), 3)
