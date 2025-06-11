from __future__ import annotations

"""Utilities for analyzing tokenized datasets."""

from typing import Optional, Dict, Any, Tuple
import argparse
import json
from collections import Counter

from datasets import Dataset

from data.dataset_loader import load_and_tokenize


def analyze_tokenized_dataset(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute statistics for a tokenized dataset including length distribution and top tokens.

    Parameters
    ----------
    dataset:
        HuggingFace ``Dataset`` object with an ``input_ids`` column.
    max_samples:
        Optional limit on the number of samples to analyze.

    Returns
    -------
    dict
        Dictionary containing sample count, average token length, vocabulary size,
        token length distribution and the most common tokens.
    """
    total_len = 0
    vocab = set()
    length_counter: Counter[int] = Counter()
    token_freq: Counter[int] = Counter()
    bigram_freq: Counter[Tuple[int, int]] = Counter()
    samples = 0

    for sample in dataset:
        ids = sample["input_ids"]
        total_len += len(ids)
        vocab.update(ids)
        length_counter[len(ids)] += 1
        token_freq.update(ids)
        # Update bigram frequency with consecutive token pairs
        bigram_freq.update(zip(ids[:-1], ids[1:]))
        samples += 1
        if max_samples is not None and samples >= max_samples:
            break

    avg_len = float(total_len / samples) if samples else 0.0
    top_tokens = [tok for tok, _ in token_freq.most_common(top_k)]
    top_bigrams = [list(bg) for bg, _ in bigram_freq.most_common(top_k)]
    lexical_diversity = float(len(vocab) / total_len) if total_len else 0.0

    return {
        "samples": samples,
        "avg_length": avg_len,
        "vocab_size": len(vocab),
        "length_distribution": dict(length_counter),
        "top_tokens": top_tokens,
        "top_bigrams": top_bigrams,
        "lexical_diversity": lexical_diversity,
    }


def analyze_dataset(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    *,
    text_column: str = "text",
    batch_size: int = 1000,
    num_proc: Optional[int] = None,
    max_length: Optional[int] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Load and analyze a dataset using ``load_and_tokenize``."""
    ds = load_and_tokenize(
        dataset_name,
        split,
        tokenizer_name,
        text_column=text_column,
        batch_size=batch_size,
        num_proc=num_proc,
        max_length=max_length,
        streaming=streaming,
    )
    return analyze_tokenized_dataset(ds, max_samples=max_samples, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tokenized dataset statistics")
    parser.add_argument("dataset_name")
    parser.add_argument("split")
    parser.add_argument("tokenizer_name")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to report")
    args = parser.parse_args()
    stats = analyze_dataset(
        args.dataset_name,
        args.split,
        args.tokenizer_name,
        text_column=args.text_column,
        max_samples=args.max_samples,
        top_k=args.top_k,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
