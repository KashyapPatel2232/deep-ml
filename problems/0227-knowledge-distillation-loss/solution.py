import torch
import torch.nn.functional as F

def distillation_loss(
	student_logits: torch.Tensor,
	teacher_logits: torch.Tensor,
	temperature: float = 1.0
) -> torch.Tensor:
	"""
	Compute knowledge distillation loss using PyTorch.
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss
	"""
	# Your code here
	student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    reduction = "sum" if student_logits.ndim == 1 else "batchmean"

    return (temperature ** 2) * F.kl_div(
        student_log_probs,
        teacher_probs.detach(),
        reduction=reduction,
    )