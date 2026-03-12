# SelfResearch/optimization/prompt_optimizer.py

from __future__ import annotations

import math
import logging
from typing import List, Optional

import torch
from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR

# --- ANSI Escape Codes for Colors and Styles (for consistent logging) ---
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

# --- Configure Logging for structured and colored output ---
class ColoredFormatter(logging.Formatter):
    FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    
    LOG_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED + Colors.BOLD,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD + Colors.UNDERLINE
    }

    def format(self, record):
        log_fmt = self.LOG_COLORS.get(record.levelno, Colors.RESET) + self.FORMAT + Colors.RESET
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    log.addHandler(console_handler)


class PromptOptimizer:
    """Generate and score prompt variations using centralized LLM wrapper."""

    def __init__(self, model_name: str = DEFAULT_GENERATOR, device: Optional[str] = None) -> None:
        """
        Initializes the PromptOptimizer.
        """
        log.info(f"{Colors.BLUE}Initializing PromptOptimizer with model: {model_name}...{Colors.RESET}")
        self.wrapper = LanguageModelWrapper(model_name=model_name, device=device)
        self.model = self.wrapper.model
        # Use wrapper's processor's tokenizer for text operations
        self.tokenizer = getattr(self.wrapper.processor, "tokenizer", self.wrapper.processor)
        self.device = self.wrapper.device

    def generate_variations(
        self, base_prompt: str, n_variations: int = 5, *, temperature: float = 1.0
    ) -> List[str]:
        """
        Generates prompt variations using batch generation for high performance.
        """
        log.info(f"{Colors.BLUE}Generating {n_variations} variations for base prompt: '{base_prompt[:80]}...'{Colors.RESET}")
        
        # Meta-prompt to guide the model to generate a variation
        prompt = f"Original Prompt: {base_prompt}\nImproved Variation:"
        
        results = self.wrapper.generate(
            prompt,
            n_variations=n_variations,
            max_new_tokens=50,
            do_sample=True,
            temperature=temperature
        )
        
        if isinstance(results, str):
            results = [results]
            
        variations = []
        for res in results:
            # Clean up the output
            cleaned = res.strip().split('\n')[0].strip()
            if cleaned:
                variations.append(cleaned)
            else:
                variations.append(base_prompt)
                
        return variations

    def score_prompt(self, prompt: str) -> float:
        """
        Scores a prompt using perplexity (lower is better/more coherent).
        """
        if self.tokenizer is None:
            log.warning("Tokenizer/Processor not available. Returning default score.")
            return 100.0
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            if hasattr(self.model, "language_model"):
                outputs = self.model.language_model(input_ids, labels=input_ids)
            else:
                outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        perplexity = math.exp(loss.item()) if loss is not None else float('nan')
        log.info(f"{Colors.CYAN}Scored prompt: perplexity={perplexity:.2f}{Colors.RESET}")
        return perplexity

    def optimize(self, base_prompt: str, n_variations: int = 5) -> str:
        """
        Main optimization loop.
        """
        variations = self.generate_variations(base_prompt, n_variations=n_variations)
        all_candidates = [base_prompt] + variations
        
        scored = []
        for p in all_candidates:
            score = self.score_prompt(p)
            if not math.isnan(score):
                scored.append((p, score))
                
        if not scored:
            return base_prompt
            
        best_prompt, best_score = min(scored, key=lambda x: x[1])
        
        log.info(f"{Colors.GREEN}Optimization complete. Best score: {best_score:.2f}{Colors.RESET}")
        return best_prompt

# Keep backward compatibility
def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
    return self.optimize(base_prompt, n_variations)

PromptOptimizer.optimize_prompt = optimize_prompt

if __name__ == "__main__":
    opt = PromptOptimizer()
    base = "Tell me about space."
    best = opt.optimize(base, n_variations=2)
    print(f"Original: {base}")
    print(f"Optimized: {best}")
