import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        batch_size , seq_len, d_model = x.shape 
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)

        attn = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
        if mask is not None:
            masked = attn + mask
        else:
            masked = attn

        scores = torch.softmax(masked, dim=-1)

        output = (scores @ V).transpose(1,2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_proj(output)