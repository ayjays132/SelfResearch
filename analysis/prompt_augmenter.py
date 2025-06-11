"""Prompt augmentation utilities."""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PromptAugmenter:
    """Generate additional prompt variations for training."""

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

    def augment_prompt(self, prompt: str, n_variations: int = 3) -> List[str]:
        """Return multiple variations of a single prompt."""
        outputs = self.generator(prompt, num_return_sequences=n_variations, max_new_tokens=30)
        variations = []
        for out in outputs:
            text = out["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            new_prompt = f"{prompt} {text}".strip()
            variations.append(new_prompt)
        return variations

    def augment_dataset(self, prompts: List[str], n_variations: int = 3) -> List[str]:
        """Augment a list of prompts with generated variations."""
        augmented: List[str] = []
        for p in prompts:
            augmented.append(p)
            augmented.extend(self.augment_prompt(p, n_variations=n_variations))
        return augmented
