from __future__ import annotations

"""Utilities for analyzing tokenized datasets."""

from typing import Optional, Iterable, Dict
import argparse
import json

from datasets import Dataset
from transformers import AutoTokenizer

from data.dataset_loader import load_and_tokenize


def analyze_tokenized_dataset(dataset: Dataset, max_samples: Optional[int] = None) -> Dict[str, float]:
    """Compute simple statistics for a tokenized dataset.

    Parameters
    ----------
    dataset:
        HuggingFace ``Dataset`` object with an ``input_ids`` column.
    max_samples:
        Optional limit on the number of samples to analyze.

    Returns
    -------
    dict
        Dictionary containing sample count, average token length and
        vocabulary size.
    """
    total_len = 0
    vocab = set()
    samples = 0
    for i, sample in enumerate(dataset):
        total_len += len(sample["input_ids"])
        vocab.update(sample["input_ids"])
        samples += 1
        if max_samples is not None and samples >= max_samples:
            break
    avg_len = float(total_len / samples) if samples else 0.0
    return {"samples": samples, "avg_length": avg_len, "vocab_size": len(vocab)}


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
) -> Dict[str, float]:
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
    return analyze_tokenized_dataset(ds, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tokenized dataset statistics")
    parser.add_argument("dataset_name")
    parser.add_argument("split")
    parser.add_argument("tokenizer_name")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    stats = analyze_dataset(
        args.dataset_name,
        args.split,
        args.tokenizer_name,
        text_column=args.text_column,
        max_samples=args.max_samples,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
