import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .embeddings import EmbeddingLayer
from .transformer import TransformerBlock


class GPTModel(nn.Module):
    """
    Tiny GPT-style language model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embed = EmbeddingLayer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T) input token IDs
        targets: (B, T) or None
        returns: logits, loss
        """
        B, T = idx.shape

        # crop to max_seq_len if needed
        if T > self.config.max_seq_len:
            idx = idx[:, -self.config.max_seq_len:]
            if targets is not None:
                targets = targets[:, -self.config.max_seq_len:]
            T = self.config.max_seq_len

        x = self.embed(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond, None)

            logits_last = logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
        return idx
