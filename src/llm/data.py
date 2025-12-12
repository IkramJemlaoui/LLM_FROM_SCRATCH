import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """
    Very simple character-level tokenizer:
    - build vocab from all characters in the text
    - encode: text -> list of ints
    - decode: list of ints -> text
    """

    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi, itos)

    @classmethod
    def from_vocab(cls, stoi: Dict[str, int], itos: Dict[str, str]) -> "CharTokenizer":
        # itos will be stored with string keys in JSON, convert to int keys
        itos_int = {int(k): v for k, v in itos.items()}
        return cls(stoi, itos_int)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path) -> "CharTokenizer":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return CharTokenizer.from_vocab(data["stoi"], data["itos"])


class CharDataset(Dataset):
    """
    Takes a long sequence of token ids and returns small chunks (blocks)
    with inputs and targets shifted by one.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # how many full blocks can we make?
        return max(len(self.data) - self.block_size, 0)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def load_text_and_tokenizer(
    input_path: Path,
    vocab_path: Path,
    val_fraction: float = 0.1,
) -> Tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """
    Loads raw text, builds tokenizer, splits into train/val,
    and saves vocab.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input text file not found: {input_path}\n"
            "Create it and put some training text inside."
        )

    text = input_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(vocab_path)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - val_fraction))
    train_data = data[:n]
    val_data = data[n:]
    return tokenizer, train_data, val_data
