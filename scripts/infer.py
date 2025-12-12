from pathlib import Path

import torch

from llm.config import GPTConfig
from llm.data import CharTokenizer
from llm.model import GPTModel
from llm.utils import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "checkpoints" / "char-gpt.pt"
    vocab_path = project_root / "data" / "processed" / "vocab.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train the model first."
        )

    tokenizer = CharTokenizer.load(vocab_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint["config"]
    config = GPTConfig(
        vocab_size=config_dict["vocab_size"],
        d_model=config_dict["d_model"],
        n_layers=config_dict["n_layers"],
        n_heads=config_dict["n_heads"],
        d_ff=config_dict["d_ff"],
        max_seq_len=config_dict["max_seq_len"],
        dropout=config_dict["dropout"],
    )

    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---- Prompt ----
    prompt_text = input("Enter a prompt: ")
    if len(prompt_text) == 0:
        prompt_text = "Hello"

    prompt_ids = tokenizer.encode(prompt_text)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=200)

    out_ids = out[0].tolist()
    generated_text = tokenizer.decode(out_ids)
    print("\n=== Generated text ===")
    print(generated_text)


if __name__ == "__main__":
    main()
