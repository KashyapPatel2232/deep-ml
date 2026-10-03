import torch

def adamw_update(w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, t: int, lr: float, beta1: float, beta2: float, epsilon: float, weight_decay: float) -> tuple:
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter tensor (torch.Tensor)
      g: gradient tensor (torch.Tensor)
      m: first moment tensor (torch.Tensor)
      v: second moment tensor (torch.Tensor)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here
    m = beta1 * m + (1-beta1)*g
    v = beta2 * v + (1-beta2)*(g**2)
    m_cap = m /(1-beta1**t)
    v_cap = v / (1-beta2**t)
    w -= lr*weight_decay*w
    w -= lr*m_cap/(torch.sqrt(v_cap) + epsilon)
    return w, m, v