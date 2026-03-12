"""Prompt optimization using Bayesian optimization and centralized model state."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

from skopt import gp_minimize
from skopt.space import Real

from models.model_wrapper import DEFAULT_GENERATOR
from .prompt_optimizer import PromptOptimizer

log = logging.getLogger(__name__)

class BayesianPromptOptimizer(PromptOptimizer):
    """
    Optimize prompts by tuning generation temperature via Bayesian search.
    Leverages ModelRegistry for shared model state.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATOR,
        *,
        iterations: int = 5,
        temp_range: Tuple[float, float] = (0.5, 1.5),
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.iterations = iterations
        self.temp_range = temp_range

    def optimize(self, base_prompt: str, n_variations: int = 5) -> str:
        """
        Return the best prompt found via Bayesian optimization of temperature.
        """
        history: List[Tuple[str, float]] = []

        def objective(params: List[float]) -> float:
            temp = params[0]
            # Use the base class generate_variations which uses the wrapper
            cands = self.generate_variations(
                base_prompt, n_variations=1, temperature=temp
            )
            cand = cands[0] if cands else base_prompt
            score = self.score_prompt(cand)
            history.append((cand, score))
            return score

        gp_minimize(
            objective,
            [Real(self.temp_range[0], self.temp_range[1])],
            n_calls=self.iterations,
            random_state=42,
        )
        
        if not history:
            return base_prompt
            
        best_prompt, _ = min(history, key=lambda x: x[1])
        return best_prompt

# Maintain backward compatibility if needed, though 'optimize' is preferred
def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
    return self.optimize(base_prompt, n_variations)

BayesianPromptOptimizer.optimize_prompt = optimize_prompt
