import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    attention = Q @ K.transpose(-2,-1) / torch.sqrt(torch.tensor(K.shape[-1]))
    masked_attn = attention + mask
    norm_masked_attn = torch.max(masked_attn, dim = 1, keepdim=True).values
    sft_attention = torch.softmax(masked_attn - norm_masked_attn, dim = -1)
    output = sft_attention @ V
    return output