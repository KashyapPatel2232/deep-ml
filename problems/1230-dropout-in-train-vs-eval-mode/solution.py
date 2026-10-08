import torch
import torch.nn as nn


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """
    # TODO: seed, ones(10), Dropout(0.5), eval output, train nonzero count
    torch.manual_seed(0)
    x = torch.ones(10)
    drop = nn.Dropout(p = 0.5)

    drop.eval()
    eval_out = drop(x)

    drop.train()
    train_out = drop(x)
    count = int(torch.sum(train_out > 0).item())

    return eval_out, count
