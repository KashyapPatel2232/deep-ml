import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate the F-beta score for a binary classification task.

    :param y_true: NumPy array of true labels
    :param y_pred: NumPy array of predicted labels
    :param beta: Non-negative weight for recall
    :return: F-beta score rounded to three decimal places
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0

    score = (1 + beta_squared) * precision * recall / denominator
    return round(score, 3)
