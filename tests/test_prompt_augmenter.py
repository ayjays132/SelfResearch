import pytest
from analysis.prompt_augmenter import PromptAugmenter
from unittest.mock import patch


def test_augment_prompt():
    aug = PromptAugmenter("distilgpt2")
    with patch.object(aug, "generator") as gen_mock:
        gen_mock.return_value = [
            {"generated_text": "base variation1"},
            {"generated_text": "base variation2"},
        ]
        variations = aug.augment_prompt("base", n_variations=2)
    assert variations == ["base variation1", "base variation2"]


def test_augment_dataset():
    aug = PromptAugmenter("distilgpt2")
    with patch.object(aug, "augment_prompt", side_effect=[["p1 v1", "p1 v2"], ["p2 v1", "p2 v2"]]):
        prompts = aug.augment_dataset(["p1", "p2"], n_variations=2)
    assert prompts == ["p1", "p1 v1", "p1 v2", "p2", "p2 v1", "p2 v2"]
