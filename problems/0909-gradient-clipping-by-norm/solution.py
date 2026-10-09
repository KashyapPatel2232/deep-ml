def clip_grad_norm(parameters, max_norm: float) -> float:
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return 0.0

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])
    )

    if total_norm.item() > max_norm:
        scale = max_norm / total_norm
        with torch.no_grad():
            for grad in grads:
                grad.mul_(scale)

    return total_norm.item()