import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    # Your code here
    idx = list(idx)
    
    for _ in range(max_new_tokens):
        if len(idx) > context_size:
            current_context = idx[- context_size:]
        else:
            current_context = idx

        current_context = np.asarray(current_context, dtype=np.int64).reshape(1, -1)
        logits = model(current_context)
        temp = int(np.argmax(logits[:, -1, :], axis = -1).item())
        idx.append(temp)
    return idx