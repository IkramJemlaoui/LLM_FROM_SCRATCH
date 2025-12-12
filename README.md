# tinyGPT from Scratch

This project implements a minimal GPT-style language model from scratch in PyTorch.
It covers tokenization, dataset construction, a decoder-only Transformer architecture,
training with an autoregressive objective, and text generation.

## Structure
- `src/llm/` – core model implementation
- `src/tinygpt/` – public API
- `scripts/` – training and inference scripts
- `data/` – input text and vocabulary

## Usage

```bash
pip install -e .
python scripts/train.py
python scripts/infer.py
