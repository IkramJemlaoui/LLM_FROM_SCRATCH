import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm.config import GPTConfig
from llm.data import CharDataset, load_text_and_tokenizer
from llm.model import GPTModel
from llm.utils import set_seed, get_device, ensure_dir


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[1]
    data_raw_path = project_root / "data" / "raw" / "input.txt"
    vocab_path = project_root / "data" / "processed" / "vocab.json"
    checkpoints_dir = project_root / "checkpoints"
    ensure_dir(checkpoints_dir)

    # ---- Load data and tokenizer ----
    tokenizer, train_data, val_data = load_text_and_tokenizer(
        input_path=data_raw_path,
        vocab_path=vocab_path,
    )

    block_size = 16
    batch_size = 32
    max_steps = 3000  # keep small for beginner / CPU
    eval_interval = 100
    learning_rate = 3e-4

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Model ----
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=block_size,
        dropout=0.1,
    )
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    step = 0
    model.train()

    while step < max_steps:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0:
                print(f"Step {step}/{max_steps} - train loss: {loss.item():.4f}")

            if step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f" --> Validation loss at step {step}: {val_loss:.4f}")

            if step >= max_steps:
                break

    # ---- Save checkpoint ----
    ckpt_path = checkpoints_dir / "char-gpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


@torch.no_grad()
def evaluate(model: GPTModel, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else math.nan


if __name__ == "__main__":
    main()
