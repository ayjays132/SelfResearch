from __future__ import annotations

"""Prompt augmentation utilities using centralized resources and Qwen-based safety logic."""

import logging
from typing import List, Optional, Any
import re

import torch
from models.model_wrapper import LanguageModelWrapper, ModelRegistry, DEFAULT_GENERATOR

# --- ANSI Escape Codes for Colors and Styles ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    INVERT = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)


class PromptAugmenter:
    """
    An advanced, algorithmic, and ethically-conscious prompt augmentation utility.
    Safety and relevance checking is now handled by Qwen 3.5 to eliminate large BART downloads.
    """

    def __init__(self, model_name: str = DEFAULT_GENERATOR, device: Optional[str] = None) -> None:
        log.info(f"{Colors.BLUE}Initializing PromptAugmenter for model: '{model_name}'...{Colors.RESET}")
        self.device_str = device
        try:
            # Shared generator (Qwen 3.5)
            self.generator = LanguageModelWrapper(model_name=model_name, device=self.device_str)
            log.info(f"{Colors.GREEN}LanguageModelWrapper initialized for prompt generation and safety logic.{Colors.RESET}")
        except Exception as e:
            log.critical(f"{Colors.RED}Failed to initialize PromptAugmenter: {e}{Colors.RESET}")
            raise RuntimeError(f"PromptAugmenter initialization failed: {e}")

    def augment(self, 
                base_prompt: str, 
                n_variations: int = 5, 
                strategy: str = "creative", 
                **kwargs: Any) -> List[str]:
        log.info(f"{Colors.BLUE}Augmenting base prompt: '{base_prompt[:50]}...' (strategy={strategy}){Colors.RESET}")
        
        params = {
            "creative": {"temperature": 1.2, "top_p": 0.95, "repetition_penalty": 1.2},
            "balanced": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1},
            "focused":  {"temperature": 0.4, "top_p": 0.85, "repetition_penalty": 1.05}
        }.get(strategy, {"temperature": 0.7})
        
        params.update(kwargs)

        meta_prompt = (f"Act as an expert prompt engineer. Generate a distinct and effective "
                       f"variation of the following prompt while preserving its original intent: "
                       f"'{base_prompt}'\n\nVariation:")

        try:
            variations_raw = []
            # Generate one by one to allow incremental safety checking without complex batching
            for _ in range(n_variations):
                var = self.generator.generate(meta_prompt, **params)
                cleaned_var = var.strip().split('\n')[0].strip()
                
                if self._is_safe_and_relevant(cleaned_var, base_prompt):
                    variations_raw.append(cleaned_var)
                    log.info(f"{Colors.GREEN}  Safe Variation: {cleaned_var}{Colors.RESET}")
                else:
                    log.warning(f"{Colors.YELLOW}  Variation rejected by Qwen safety/relevance logic.{Colors.RESET}")

            return variations_raw

        except Exception as e:
            log.error(f"{Colors.RED}Prompt augmentation failed: {e}{Colors.RESET}")
            return []

    def _is_safe_and_relevant(self, text: str, original_prompt: str) -> bool:
        """
        Uses Qwen 3.5 to perform safety and relevance classification.
        Replaces BART zero-shot to save memory.
        """
        if not text or len(text) < 5: return False
        
        # Classification prompt
        classification_prompt = (
            f"Analyze the following text for safety and relevance to the original goal.\n"
            f"Original Goal: '{original_prompt}'\n"
            f"Candidate Variation: '{text}'\n\n"
            f"Determine if the variation is SAFE (non-toxic, non-hateful) and RELEVANT (same intent).\n"
            f"Respond exactly with 'YES' if it is both safe and relevant, otherwise 'NO'.\n"
            f"Decision:"
        )
        
        decision = self.generator.generate(classification_prompt, max_new_tokens=5, temperature=0.1)
        return "YES" in decision.upper()

if __name__ == "__main__":
    try:
        augmenter = PromptAugmenter()
        base = "Research quantum computing impacts on medicine."
        vars = augmenter.augment(base, n_variations=2)
        print(f"Base: {base}")
        print(f"Variations: {vars}")
    except Exception as e:
        log.error(f"Demo failed: {e}")
