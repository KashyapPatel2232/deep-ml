import torch

def learned_positional_encoding(token_embeddings: torch.Tensor, position_embedding_table: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Apply learned positional embeddings to token embeddings.
    
    Args:
        token_embeddings: (batch_size, seq_len, d_model) tensor of token embeddings
        position_embedding_table: (max_seq_len, d_model) learned positional embedding lookup table
        start_pos: Starting position index (default 0)
    
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) with positional information applied
    """
    B, seq_len, d_model = token_embeddings.shape
    max_seq_len, d_model = position_embedding_table.shape
    min_len = min(seq_len, max_seq_len)
    output = token_embeddings + position_embedding_table[start_pos: start_pos + min_len, :]
    return output