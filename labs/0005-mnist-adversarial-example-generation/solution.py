import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate a strong adversarial example for input x using Momentum PGD + Margin Loss.
    '''
    x_orig = x.detach()
    if epsilon <= 0:
        return x_orig.clone()

    y_true = torch.as_tensor(y, device=x.device, dtype=torch.long).view(-1)
    batch_size = x_orig.size(0)

    eps = max(0.0, float(epsilon) - 1e-6)
    lower_bound = (x_orig - eps).clamp(0.0, 1.0)
    upper_bound = (x_orig + eps).clamp(0.0, 1.0)

    num_steps = 40
    num_restarts = 5

    base_step_size = eps / 4.0
    momentum_decay = 1.0

    best_x = x_orig.clone()

    best_margin = torch.full((batch_size,), float("inf"), device=x.device)

    with torch.enable_grad():
        for restart in range(num_restarts):
            if restart == 0:
                x_adv = x_orig.clone()
            else:

                delta = torch.empty_like(x_orig).uniform_(-eps, eps)
                x_adv = torch.max(torch.min(x_orig + delta, upper_bound), lower_bound)

            momentum = torch.zeros_like(x_orig)

            for step in range(num_steps):
                x_adv.requires_grad_(True)
                logits = model(x_adv)

                true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
                
                logits_masked = logits.clone()
                logits_masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
                other_logits = logits_masked.max(dim=1).values

                margin = true_logits - other_logits


                improved = margin < best_margin
                if improved.any():
                    best_margin = torch.where(improved, margin.detach(), best_margin)
                    mask = improved.view(-1, *([1] * (x_orig.ndim - 1)))
                    best_x = torch.where(mask, x_adv.detach(), best_x)

                
                ce_loss = criterion(logits, y_true)
                if ce_loss.ndim > 0:
                    ce_loss = ce_loss.mean()
                
                total_loss = -margin.sum() + ce_loss

                grad = torch.autograd.grad(total_loss, x_adv)[0]


                grad_norm = grad.abs().flatten(1).mean(dim=1).view(-1, *([1] * (x_orig.ndim - 1))) + 1e-12
                momentum = momentum_decay * momentum + (grad / grad_norm)

                
                curr_step = base_step_size * 0.5 * (1.0 + torch.cos(torch.tensor(step / num_steps * 3.14159265, device=x.device)))

                x_adv = x_adv.detach() + curr_step * momentum.sign()
                x_adv = torch.max(torch.min(x_adv, upper_bound), lower_bound)

        with torch.no_grad():
            logits = model(x_adv)
            true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
            logits_masked = logits.clone()
            logits_masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
            other_logits = logits_masked.max(dim=1).values
            margin = true_logits - other_logits

            improved = margin < best_margin
            if improved.any():
                mask = improved.view(-1, *([1] * (x_orig.ndim - 1)))
                best_x = torch.where(mask, x_adv.detach(), best_x)

    return best_x.detach()