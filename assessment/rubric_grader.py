import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Any, Optional
import re

from models.model_wrapper import ModelRegistry, LanguageModelWrapper, DEFAULT_GENERATOR, DEFAULT_EMBEDDER

log = logging.getLogger(__name__)

class RubricGrader:
    """
    An advanced, automated rubric-based grader for textual research proposals/reports.
    Rewritten to use centralized ModelRegistry (Qwen 3.5 & Gemma Embeddings)
    to save memory and run much faster without duplicate models.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = ModelRegistry.get_device(device)
        log.info(f"RubricGrader initialized on {self.device}")
        
        # Load LLM (which now handles embedding dispatch)
        self.llm = LanguageModelWrapper(model_name=DEFAULT_GENERATOR, device=str(self.device))
        
        self._rubric_embeddings_cache: Dict[str, torch.Tensor] = {}

    def _load_rubric_embeddings(self, rubric: Dict[str, Any]):
        log.info("Loading and caching rubric embeddings...")
        new_cache: Dict[str, torch.Tensor] = {}
        for criterion_name, criterion_details in rubric.items():
            expected_content = criterion_details.get('expected_content', '')
            if expected_content:
                # Use the LLM's embed method (Local or Gemini 2.0)
                vec = self.llm.embed(expected_content)
                if vec:
                    new_cache[criterion_name] = torch.tensor(vec).to(self.device)
        self._rubric_embeddings_cache = new_cache
        log.info(f"Rubric embeddings cached for {len(self._rubric_embeddings_cache)} criteria.")

    def _grade_criterion(self, submission_text: str, criterion_name: str, criterion_details: Dict[str, Any]) -> Dict[str, Any]:
        expected_content = criterion_details.get('expected_content', '')
        max_score = criterion_details.get('max_score', 0)

        if not expected_content or max_score <= 0:
            return {"score": 0.0, "max_score": max_score, "similarity": 0.0, "feedback": "No content."}

        expected_embedding = self._rubric_embeddings_cache.get(criterion_name)
        if expected_embedding is None:
            vec = self.llm.embed(expected_content)
            expected_embedding = torch.tensor(vec).to(self.device)

        sub_vec = self.llm.embed(submission_text)
        submission_embedding = torch.tensor(sub_vec).to(self.device)

        similarity = cosine_similarity(
            submission_embedding.cpu().numpy().reshape(1, -1), 
            expected_embedding.cpu().numpy().reshape(1, -1)
        )[0][0]
        
        similarity = max(0.0, min(1.0, float(similarity)))
        score = similarity * max_score
        
        # Generate feedback using Qwen
        prompt = (
            f"As an expert academic reviewer, provide an in-depth critique of this submission snippet against the expected criteria.\n"
            f"Submission: {submission_text[:600]}...\n"
            f"Expected: {expected_content[:600]}\n"
            f"Score calculated: {score:.2f}/{max_score}.\n"
            f"Provide a structured response with:\n"
            f"1. **Strengths**: What they did right.\n"
            f"2. **Weaknesses**: What is missing compared to the expected criteria.\n"
            f"3. **Actionable Steps**: Exactly how to improve.\n"
            f"Feedback:"
        )
        feedback = self.llm.generate(prompt, max_new_tokens=250, temperature=0.6)
        
        return {
            "score": score,
            "max_score": max_score,
            "similarity": similarity,
            "feedback": feedback.strip()
        }

    def _calculate_overall_summary(self, grades: Dict[str, Any], rubric: Dict[str, Any]) -> Dict[str, Any]:
        total_score = sum(result.get('score', 0.0) for result in grades.values() if isinstance(result, dict) and 'score' in result)
        max_total_score = sum(crit.get('max_score', 0) for crit in rubric.values() if isinstance(crit, dict))
        percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0.0

        snippets = []
        for c, g in grades.items():
            if c != "overall_summary" and isinstance(g, dict):
                snippets.append(f"{c}: {g.get('score', 0):.1f}/{g.get('max_score', 0)} - {g.get('feedback', '')}")
        
        prompt = (
            f"Based on these scores:\n" + "\n".join(snippets) + "\n"
            f"Total: {total_score:.1f}/{max_total_score} ({percentage:.1f}%).\n"
            f"Provide a 2-sentence empathetic overall summary and recommendation.\nSummary:"
        )
        overall_feedback = self.llm.generate(prompt, max_new_tokens=150, temperature=0.6).strip()

        return {
            "total_score": total_score,
            "max_total_score": max_total_score,
            "percentage": percentage,
            "overall_feedback": overall_feedback
        }

    def grade_submission(self, submission_text: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        self._load_rubric_embeddings(rubric)
        results = {}
        for criterion_name, criterion_details in rubric.items():
            results[criterion_name] = self._grade_criterion(submission_text, criterion_name, criterion_details)
        results["overall_summary"] = self._calculate_overall_summary(results, rubric)
        return results

    def grade_batch(self, submissions: List[str], rubric: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._load_rubric_embeddings(rubric)
        return [self.grade_submission(sub, rubric) for sub in submissions]
