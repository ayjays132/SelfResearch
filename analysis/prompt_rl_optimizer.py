from __future__ import annotations
"""Reinforcement learning based prompt optimizer."""

import random
from typing import Callable, Dict, Optional

from .prompt_optimizer import PromptOptimizer


class PromptRLOptimizer(PromptOptimizer):
    """Refine prompts using a simple Q-learning algorithm."""

    def __init__(
        self,
        model_name: str,
        reward_fn: Callable[[str], float] | None = None,
        *,
        episodes: int = 10,
        epsilon: float = 0.2,
        lr: float = 0.3,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.reward_fn = reward_fn
        self.episodes = episodes
        self.epsilon = epsilon
        self.lr = lr
        self._q_values: Dict[str, float] = {}

    def _reward(self, prompt: str) -> float:
        if self.reward_fn is not None:
            return self.reward_fn(prompt)
        return -self.score_prompt(prompt)

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        for c in candidates:
            if c not in self._q_values:
                self._q_values[c] = self._reward(c)

        for _ in range(self.episodes):
            if random.random() < self.epsilon:
                prompt = random.choice(candidates)
            else:
                prompt = max(candidates, key=lambda p: self._q_values.get(p, 0.0))
            reward = self._reward(prompt)
            self._q_values[prompt] = self._q_values.get(prompt, 0.0) + self.lr * (
                reward - self._q_values.get(prompt, 0.0)
            )
        best_prompt = max(self._q_values, key=self._q_values.get)
        return best_prompt
