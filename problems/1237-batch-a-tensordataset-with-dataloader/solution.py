import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    batches = DataLoader(dataset=dataset, batch_size= 4, shuffle=False)
    first_batch_X, _ = next(iter(batches))

    return len(batches), tuple(first_batch_X.shape)