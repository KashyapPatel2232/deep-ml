def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    if not isinstance(y_true, (list, tuple)):
        y_true = [y_true]
    if not isinstance(y_pred, (list, tuple)):
        y_pred = [y_pred]

    total = 0.0
    n = len(y_true)
    
    for yt, yp in zip(y_true, y_pred):
        error = abs(yt - yp)
        if error <= delta:
            total += 0.5 * (error ** 2)
        else:
            total += delta * (error - 0.5 * delta)

    return total / n if n > 0 else 0.0