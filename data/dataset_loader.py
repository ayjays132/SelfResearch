
"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def load_and_tokenize(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    text_column: str = "text",
    batch_size: int = 1000,
) -> Dataset:
    """Load a dataset from the HuggingFace Hub and tokenize it.

    Args:
        dataset_name: Name or path of the dataset to load (e.g. ``"ag_news"``).
        split: Which split to load (e.g. ``"train[:1000]"``).
        tokenizer_name: Name of the tokenizer/model to tokenize with.
        text_column: Name of the text column to tokenize.
        batch_size: Batch size for tokenization.

    Returns:
        The tokenized dataset formatted with PyTorch tensors.
    """
    dataset = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch: Any) -> Any:
        return tokenizer(batch[text_column], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize, batched=True, batch_size=batch_size)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def load_dataset_splits(
    dataset_name: str,
    tokenizer_name: str,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    text_column: str = "text",
    batch_size: int = 1000,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and tokenize train/eval dataset splits.

    Args:
        dataset_name: Dataset name or local path.
        tokenizer_name: Pretrained tokenizer name.
        train_split: Split for training data.
        eval_split: Optional evaluation split.
        text_column: Name of the text column to tokenize.
        batch_size: Batch size for tokenization.

    Returns:
        A tuple of ``(train_dataset, eval_dataset)`` where ``eval_dataset`` may
        be ``None`` if ``eval_split`` is not provided.
    """
    train_ds = load_and_tokenize(
        dataset_name,
        train_split,
        tokenizer_name,
        text_column=text_column,
        batch_size=batch_size,
    )
    eval_ds = None
    if eval_split:
        eval_ds = load_and_tokenize(
            dataset_name,
            eval_split,
            tokenizer_name,
            text_column=text_column,
            batch_size=batch_size,
        )
    return train_ds, eval_ds


if __name__ == "__main__":
    ds, _ = load_dataset_splits("ag_news", "distilgpt2", train_split="train[:10]")
    print(ds[0])
