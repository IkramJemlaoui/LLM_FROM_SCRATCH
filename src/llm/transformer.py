import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Simple Transformer block:
    - LayerNorm
    - Multi-head attention + residual
    - LayerNorm
    - Feedforward MLP + residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attention block
        x = x + self.dropout(self.attn(self.ln1(x)))
        # feed-forward block
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
