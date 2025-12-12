import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Token embedding + positional embedding.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time)
        returns: (batch, time, d_model)
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)[None, :, :]
        return token_embeddings + position_embeddings
