import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    # Your code here
    rng = np.random.default_rng(seed)
    batch_start_id = rng.integers(0, len(data) - block_size, size = batch_size)
    batch = [[],[]]
    for offset in batch_start_id:
        x = data[offset: offset + block_size].tolist()
        y = data[offset + 1: offset + 1 + block_size].tolist()
        batch[0].append(x)
        batch[1].append(y)

    return np.asarray(batch)