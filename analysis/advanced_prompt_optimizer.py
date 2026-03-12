"""Advanced prompt optimization with Persistent RAG semantic similarity."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from .prompt_optimizer import PromptOptimizer
from models.model_wrapper import DEFAULT_GENERATOR
from memory.persistent_rag import PersistentRAG


class AdvancedPromptOptimizer(PromptOptimizer):
    """Optimize prompts using perplexity and semantic RAG similarity."""

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATOR,
        *,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        similarity_weight: float = 0.5,
    ) -> None:
        super().__init__(model_name, device=device)
        self.rag = PersistentRAG(device=device)
        self.similarity_weight = similarity_weight

    def _embedding(self, text: str) -> torch.Tensor:
        # Use RAG's built in SentenceTransformer instead of raw HuggingFace embedder
        return self.rag.embedder.encode(text, convert_to_tensor=True).to(self.device)

    def score_prompt(self, prompt: str, *, base_embedding: torch.Tensor) -> float:
        perplexity = super().score_prompt(prompt)
        emb = self._embedding(prompt)
        similarity = F.cosine_similarity(emb, base_embedding, dim=0).item()
        return perplexity - self.similarity_weight * similarity

    def optimize(self, base_prompt: str, n_variations: int = 5) -> str:
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        all_candidates = [base_prompt] + candidates
        base_emb = self._embedding(base_prompt)
        
        best_prompt: str = all_candidates[0]
        best_score = float("inf")
        for cand in all_candidates:
            score = self.score_prompt(cand, base_embedding=base_emb)
            if score < best_score:
                best_score = score
                best_prompt = cand
        return best_prompt

# Keep backward compatibility 
def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
    return self.optimize(base_prompt, n_variations)
AdvancedPromptOptimizer.optimize_prompt = optimize_prompt
