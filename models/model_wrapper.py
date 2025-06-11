from __future__ import annotations

"""Model wrapper utilities for HuggingFace models."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModelWrapper:
    """Simple wrapper around a causal language model."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
