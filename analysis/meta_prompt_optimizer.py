"""Combine multiple prompt optimizers for enhanced results."""

from __future__ import annotations

from typing import Optional

from .prompt_optimizer import PromptOptimizer
from .prompt_bandit_optimizer import PromptBanditOptimizer
from .prompt_annealing_optimizer import PromptAnnealingOptimizer
from .prompt_rl_optimizer import PromptRLOptimizer


class MetaPromptOptimizer(PromptOptimizer):
    """Run several optimizers sequentially and return the best prompt."""

    def __init__(self, model_name: str, *, device: Optional[str] = None) -> None:
        super().__init__(model_name, device=device)
        # Use internal score_prompt as reward for bandit and RL optimizers
        self.bandit = PromptBanditOptimizer(
            model_name,
            reward_fn=lambda p: -self.score_prompt(p),
            device=device,
            iterations=3,
            epsilon=0.2,
        )
        self.annealer = PromptAnnealingOptimizer(model_name, device=device)
        self.rl = PromptRLOptimizer(
            model_name,
            reward_fn=lambda p: -self.score_prompt(p),
            device=device,
            episodes=3,
            epsilon=0.1,
        )

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """Apply each optimizer and choose the overall best prompt."""
        prompt1 = self.bandit.optimize_prompt(base_prompt, n_variations=n_variations)
        prompt2 = self.annealer.optimize_prompt(prompt1, n_variations=n_variations)
        prompt3 = self.rl.optimize_prompt(prompt2, n_variations=n_variations)
        candidates = [base_prompt, prompt1, prompt2, prompt3]
        best_prompt = min(candidates, key=self.score_prompt)
        return best_prompt
