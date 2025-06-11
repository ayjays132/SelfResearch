from unittest.mock import patch
from datasets import Dataset

from analysis.dataset_analyzer import analyze_tokenized_dataset, analyze_dataset


def test_analyze_tokenized_dataset():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [2, 3, 4, 5]]})
    stats = analyze_tokenized_dataset(ds)
    assert stats["samples"] == 2
    assert stats["avg_length"] == 3.5
    assert stats["vocab_size"] == 5


def test_analyze_dataset_patch():
    ds = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5]]})
    with patch("analysis.dataset_analyzer.load_and_tokenize", return_value=ds):
        stats = analyze_dataset("dummy", "train", "dummy")
    assert stats["samples"] == 2
    assert stats["vocab_size"] == 5
