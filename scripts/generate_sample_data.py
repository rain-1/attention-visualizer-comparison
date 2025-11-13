"""Generate toy attention tensors for local development."""

from __future__ import annotations

from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = ["<bos>", "User", ":", "Hello", ",", "Assistant", ":", "Hi", "!", "<eos>"]
NUM_LAYERS = 12
NUM_HEADS = 8


def make_attention(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Create a positive tensor and normalize rows so values resemble probabilities.
    tensor = torch.rand((NUM_LAYERS, NUM_HEADS, len(TOKENS), len(TOKENS)), generator=generator)
    tensor = tensor / tensor.sum(dim=-1, keepdim=True)
    return tensor


def save_bundle(path: Path, seed: int) -> None:
    attention = make_attention(seed)
    payload = {
        "tokens": TOKENS,
        "attention": attention,
        "prompt": "Sample conversation",
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
    }
    torch.save(payload, path)
    print(f"Wrote {path}")


def main() -> None:
    save_bundle(DATA_DIR / "model_a.pt", seed=1)
    save_bundle(DATA_DIR / "model_b.pt", seed=2)


if __name__ == "__main__":
    main()
