import numpy as np

def greedy_stream(model, prompt_ids, max_new_tokens: int, eos_id: int) -> list:
    logits = model.prefill(prompt_ids)
    if max_new_tokens <= 0:
        return []

    output = []
    for _ in range(max_new_tokens):
        next_token = int(np.argmax(logits))
        if next_token == eos_id:
            break
        output.append(next_token)
        logits = model.decode_step(next_token)

    return output