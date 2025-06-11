from typing import Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def load_and_tokenize(dataset_name: str, split: str, tokenizer_name: str, batch_size: int = 1000) -> Dataset:
    """Load a dataset from HuggingFace Hub and tokenize it using a specified tokenizer.

    Args:
        dataset_name: Name of the dataset to load (e.g., "ag_news").
        split: Which split to load (e.g., "train[:1000]").
        tokenizer_name: Name of the tokenizer/model to tokenize with.
        batch_size: Batch size for tokenization.

    Returns:
        A tokenized dataset formatted with PyTorch tensors.
    """
    dataset = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch: Any) -> Any:
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize, batched=True, batch_size=batch_size)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized

if __name__ == "__main__":
    ds = load_and_tokenize("ag_news", "train[:10]", "distilgpt2")
    print(ds[0])
