from unittest.mock import patch
from datasets import Dataset

from data.dataset_loader import load_and_tokenize


class DummyTokenizer:
    def __call__(self, text, truncation=True, padding="max_length"):
        if isinstance(text, list):
            return {"input_ids": [[1, 2]] * len(text), "attention_mask": [[1, 1]] * len(text)}
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


def test_load_and_tokenize_patch():
    dummy_ds = Dataset.from_dict({"text": ["hello", "world"]})
    with patch("data.dataset_loader.load_dataset", return_value=dummy_ds), \
         patch("data.dataset_loader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
        tokenized = load_and_tokenize("dummy", "train", "dummy")
    assert len(tokenized) == 2
    assert "input_ids" in tokenized.column_names
