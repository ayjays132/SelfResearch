"""Utilities for prompt engineering and optimization."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PromptOptimizer:
    """Generate and score prompt variations to select high quality prompts."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
        )

    def generate_variations(
        self, base_prompt: str, n_variations: int = 5, *, temperature: float = 1.0
    ) -> List[str]:
        """Generate prompt variations from a base prompt.

        Parameters
        ----------
        base_prompt:
            Prompt used as the starting point for generation.
        n_variations:
            Number of variations to produce.
        temperature:
            Sampling temperature controlling randomness of generations.
        """
        outputs = self.generator(
            base_prompt,
            num_return_sequences=n_variations,
            max_new_tokens=20,
            do_sample=True,
            temperature=temperature,
        )
        variations = []
        for out in outputs:
            text = out["generated_text"]
            if text.startswith(base_prompt):
                text = text[len(base_prompt):].strip()
            if text:
                variations.append(base_prompt + " " + text)
            else:
                variations.append(base_prompt)
        return variations

    def score_prompt(self, prompt: str) -> float:
        """Compute perplexity of a prompt as a quality score (lower is better)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        return float(math.exp(loss.item()))

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """Return the variation with the lowest perplexity score."""
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        best_prompt = min(candidates, key=self.score_prompt)
        return best_prompt
