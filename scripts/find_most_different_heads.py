"""Utility for ranking attention heads by difference between two models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class AttentionBundle:
    tokens: List[str]
    attention: torch.Tensor


def load_bundle(path: Path) -> AttentionBundle:
    payload = torch.load(path, map_location="cpu")
    if "attention" not in payload:
        raise KeyError(f"Attention tensor missing in {path}")
    attention = payload["attention"].float()
    if attention.ndim != 4:
        raise ValueError("Attention tensor must have shape [layers, heads, tokens, tokens]")
    tokens = list(payload.get("tokens") or [str(i) for i in range(attention.shape[-1])])
    return AttentionBundle(tokens=tokens, attention=attention)


def build_identity_mapping(length: int) -> List[List[int]]:
    return [[idx] for idx in range(length)]


def build_alignment(source_tokens: Sequence[str], target_tokens: Sequence[str]) -> List[List[int]]:
    """Group source token indices so that they align with target tokens.

    The algorithm keeps tokens that appear in both sequences aligned and assigns
    unmatched source tokens to the nearest target token that follows them (or the
    previous one if no such token exists).
    """

    from difflib import SequenceMatcher

    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    mapping: List[List[int]] = [[] for _ in range(len(target_tokens))]

    source_pos = 0
    for block in matcher.get_matching_blocks():
        unmatched_end = block.a
        target_insert_pos = block.b
        while source_pos < unmatched_end:
            if target_insert_pos < len(mapping):
                mapping[target_insert_pos].append(source_pos)
            elif target_insert_pos > 0:
                mapping[target_insert_pos - 1].append(source_pos)
            else:
                raise ValueError("Unable to align source tokens with target tokens")
            source_pos += 1

        for offset in range(block.size):
            target_idx = block.b + offset
            source_idx = block.a + offset
            mapping[target_idx].append(source_idx)
        source_pos = block.a + block.size

    all_assigned = sorted(index for group in mapping for index in group)
    expected = list(range(len(source_tokens)))
    if all_assigned != expected:
        raise ValueError("Alignment failed to cover every source token")

    return mapping


def compress_attention(attention: torch.Tensor, mapping: Sequence[Sequence[int]]) -> torch.Tensor:
    """Average rows/columns of ``attention`` according to ``mapping``."""

    if attention.shape[-1] != attention.shape[-2]:
        raise ValueError("Attention tensor must have square token dimensions")

    src_len = attention.shape[-1]
    covered = sorted(index for group in mapping for index in group)
    if covered != list(range(src_len)):
        raise ValueError("Mapping must cover every source token index exactly once")

    target_len = len(mapping)
    device = attention.device
    dtype = attention.dtype
    aggregated = torch.empty((*attention.shape[:2], target_len, target_len), dtype=dtype, device=device)

    index_tensors = [torch.tensor(indices, dtype=torch.long, device=device) for indices in mapping]

    for row_idx, row_indices in enumerate(index_tensors):
        row_selected = attention.index_select(2, row_indices)
        for col_idx, col_indices in enumerate(index_tensors):
            sub_tensor = row_selected.index_select(3, col_indices)
            aggregated[:, :, row_idx, col_idx] = sub_tensor.mean(dim=(2, 3))

    return aggregated


def compute_difference_scores(attn_a: torch.Tensor, attn_b: torch.Tensor) -> torch.Tensor:
    if attn_a.shape != attn_b.shape:
        raise ValueError("Attention tensors must share the same shape for comparison")
    return (attn_a - attn_b).abs().mean(dim=(-1, -2))


def _canonicalize_token(token: str) -> str:
    """Normalize ``token`` to make it easier to match the word "flowers"."""

    # Common tokenizers (GPT-2, sentencepiece, tiktoken) prefix word-start tokens with
    # characters such as ``Ġ`` or ``▁``.  We strip them before matching.
    stripped = token.strip()
    for prefix in ("Ġ", "▁", "Ċ"):
        while stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]

    # Remove typical BPE/wordpiece suffix markers.
    for suffix in ("</w>", "▁", "Ġ"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]

    # Trim surrounding punctuation so tokens like "flowers," still match.
    stripped = stripped.strip("'\".,!?;:-()").lower()
    return stripped


def _find_flowers_token_index(tokens: Sequence[str]) -> int | None:
    """Return the index of the token (or first sub-token) for "flowers" if present."""

    canonical = [_canonicalize_token(token) for token in tokens]

    # First look for an exact token match.
    for idx, token in enumerate(canonical):
        if token in {"flowers", "flower"}:
            return idx

    # Fall back to detecting multi-token splits that concatenate to ``flowers``.
    target = "flowers"
    for start in range(len(tokens)):
        combined = ""
        for end in range(start, min(len(tokens), start + len(target))):
            part = canonical[end]
            if not part:
                break
            combined += part
            if combined == target:
                return start
            if not target.startswith(combined):
                break

    return None


def describe_flowers_column_bias(
    layer_attn_a: torch.Tensor, layer_attn_b: torch.Tensor, tokens: Sequence[str]
) -> str:
    """Determine which model focuses more on the "flowers" column."""

    flowers_index = _find_flowers_token_index(tokens)
    if flowers_index is None:
        return "flowers column unavailable"

    col_mean_a = layer_attn_a[:, flowers_index].mean()
    col_mean_b = layer_attn_b[:, flowers_index].mean()

    diff = float(col_mean_a.item() - col_mean_b.item())
    tolerance = 1e-5
    if abs(diff) <= tolerance:
        return "the flowers column is similar"
    if diff > 0:
        return "the flowers column in A is hotter"
    return "the flowers column in B is hotter"


def format_top_differences(
    scores: torch.Tensor,
    attn_a: torch.Tensor,
    attn_b: torch.Tensor,
    tokens: Sequence[str],
    top_k: int,
) -> List[str]:
    layer_count, head_count = scores.shape
    flat_scores = scores.view(-1)
    values, indices = torch.topk(flat_scores, k=min(top_k, flat_scores.numel()))
    lines = ["Top differing layer/head pairs (by mean absolute difference):"]
    for rank, (value, flat_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        layer = flat_index // head_count
        head = flat_index % head_count
        descriptor = describe_flowers_column_bias(attn_a[layer, head], attn_b[layer, head], tokens)
        lines.append(
            f"{rank:2d}. Layer {layer:02d}, Head {head:02d}: {value:.6f}, {descriptor}"
        )
    return lines


def ensure_mapping(mapping: Sequence[Sequence[int]], source_length: int) -> None:
    flattened = [idx for group in mapping for idx in group]
    if sorted(flattened) != list(range(source_length)):
        raise ValueError("Mapping must cover every source index exactly once")


def build_mappings(bundle_a: AttentionBundle, bundle_b: AttentionBundle) -> tuple[List[List[int]], List[List[int]], List[str]]:
    len_a = len(bundle_a.tokens)
    len_b = len(bundle_b.tokens)

    if len_a == len_b:
        tokens = bundle_a.tokens
        mapping_a = build_identity_mapping(len_a)
        mapping_b = build_identity_mapping(len_b)
    elif len_a < len_b:
        tokens = bundle_a.tokens
        mapping_a = build_identity_mapping(len_a)
        mapping_b = build_alignment(bundle_b.tokens, tokens)
    else:
        tokens = bundle_b.tokens
        mapping_a = build_alignment(bundle_a.tokens, tokens)
        mapping_b = build_identity_mapping(len_b)

    ensure_mapping(mapping_a, len(bundle_a.tokens))
    ensure_mapping(mapping_b, len(bundle_b.tokens))

    return mapping_a, mapping_b, tokens


def categorize_alignment(mapping_a: Sequence[Sequence[int]], mapping_b: Sequence[Sequence[int]]) -> List[str]:
    categories = []
    for group_a, group_b in zip(mapping_a, mapping_b):
        len_a = len(group_a)
        len_b = len(group_b)
        if len_a > 1 and len_b > 1:
            categories.append("squashed_in_both")
        elif len_a > 1:
            categories.append("squashed_in_a_only")
        elif len_b > 1:
            categories.append("squashed_in_b_only")
        else:
            categories.append("unsquashed")
    return categories


def analyze_squashed_columns(
    attn_a: torch.Tensor,
    attn_b: torch.Tensor,
    mapping_a: Sequence[Sequence[int]],
    mapping_b: Sequence[Sequence[int]],
    tokens: Sequence[str],
) -> List[str]:
    column_mean_a = attn_a.mean(dim=(0, 1, 2))
    column_mean_b = attn_b.mean(dim=(0, 1, 2))
    column_abs_diff = (column_mean_a - column_mean_b).abs()

    categories = categorize_alignment(mapping_a, mapping_b)
    stats: Dict[str, Dict[str, float]] = {}
    total_abs_diff = float(column_abs_diff.sum().item())

    for idx, category in enumerate(categories):
        bucket = stats.setdefault(
            category,
            {"count": 0.0, "mean_a": 0.0, "mean_b": 0.0, "abs_diff": 0.0},
        )
        bucket["count"] += 1.0
        bucket["mean_a"] += float(column_mean_a[idx].item())
        bucket["mean_b"] += float(column_mean_b[idx].item())
        bucket["abs_diff"] += float(column_abs_diff[idx].item())

    lines = ["Column-level analysis (mean attention across layers/heads/queries):"]
    for category in (
        "squashed_in_a_only",
        "squashed_in_b_only",
        "squashed_in_both",
        "unsquashed",
    ):
        data = stats.get(category)
        if not data or data["count"] == 0:
            continue
        count = int(data["count"])
        mean_a = data["mean_a"] / data["count"]
        mean_b = data["mean_b"] / data["count"]
        abs_diff = data["abs_diff"]
        share = (abs_diff / total_abs_diff * 100.0) if total_abs_diff > 0 else 0.0
        lines.append(
            "- {category} (count={count}): mean_A={mean_a:.6f}, "
            "mean_B={mean_b:.6f}, |Δ|={abs_diff:.6f} ({share:.2f}% of total)".format(
                category=category,
                count=count,
                mean_a=mean_a,
                mean_b=mean_b,
                abs_diff=abs_diff,
                share=share,
            )
        )

    if total_abs_diff == 0:
        lines.append("- No column-wise differences detected (total absolute difference is zero).")

    lines.append("")
    lines.append("Token categories:")
    for token, category in zip(tokens, categories):
        lines.append(f"  {token}: {category}")

    return lines


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", type=Path, help="Path to the first attention bundle (.pt file)")
    parser.add_argument("model_b", type=Path, help="Path to the second attention bundle (.pt file)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of layer/head pairs to display")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    bundle_a = load_bundle(args.model_a)
    bundle_b = load_bundle(args.model_b)

    mapping_a, mapping_b, tokens = build_mappings(bundle_a, bundle_b)

    attn_a = compress_attention(bundle_a.attention, mapping_a)
    attn_b = compress_attention(bundle_b.attention, mapping_b)

    scores = compute_difference_scores(attn_a, attn_b)
    for line in format_top_differences(scores, attn_a, attn_b, tokens, args.top_k):
        print(line)

    print()
    print(f"Aligned token sequence length: {len(tokens)}")
    print("Tokens:")
    print(" ".join(tokens))

    print()
    for line in analyze_squashed_columns(attn_a, attn_b, mapping_a, mapping_b, tokens):
        print(line)


if __name__ == "__main__":
    main()
