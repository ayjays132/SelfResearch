import torch
import time
import threading
import logging
import copy
import random
from typing import List, Dict, Any, Optional
from rich.console import Console
from models.model_wrapper import LanguageModelWrapper
from assessment.rubric_grader import RubricGrader

log = logging.getLogger(__name__)
console = Console()

class GeneticTestTimeOptimizer:
    """
    SOTA Parallel Genetic Scaffolding.
    Mutates model layers during research using Evolutionary Strategies (ES) and Momentum.
    Evaluates mutations against a dynamic RubricGrader to ensure infinite scaling and scientific rigor.
    """
    def __init__(self, vlm_wrapper: LanguageModelWrapper, grader: RubricGrader, os_instance=None):
        self.vlm = vlm_wrapper
        self.grader = grader
        self.os = os_instance
        
        # SOTA Config
        self.mutation_scale = 1e-4
        self.momentum_beta = 0.9
        self.direction_vector = None # Momentum for successful mutations
        self.best_score = 0.0
        self.generation = 0
        self.convergence_patience = 0
        self._running = False
        self.thread = None
        
        # Evaluation Diversity
        self.eval_prompts = [
            "Explain the scientific method concisely.",
            "What is the importance of peer review in research?",
            "How does a double-blind study reduce bias?",
            "Explain the concept of statistical significance.",
            "Describe the relationship between entropy and information.",
            "How do neural networks approximate functions?",
            "Explain the importance of data normalization in machine learning."
        ]
        
        # Dynamic Rubric (Removes hard cap of 20)
        self.base_rubric = {
            "Coherency": {"expected_content": "Clear, logical scientific reasoning.", "max_score": 10},
            "Insight": {"expected_content": "Demonstrates depth of understanding.", "max_score": 10},
            "Technical Precision": {"expected_content": "Uses correct terminology and precise language.", "max_score": 10},
            "Syntactic Elegance": {"expected_content": "Fluent, professional academic tone.", "max_score": 10}
        }
        # Initial max is 40. We can add more categories as we improve.
        
        try:
            # Targeted mutation of the transformer output head for systemic behavioral shift
            self.layer_to_mutate = self.vlm.model.get_output_embeddings()
            if self.layer_to_mutate is None:
                self.layer_to_mutate = self.vlm.model.get_input_embeddings()
        except AttributeError:
            # Fallback for different model architectures
            self.layer_to_mutate = list(self.vlm.model.parameters())[-1] 
            
    def start(self):
        if not self._running:
            self._running = True
            self.thread = threading.Thread(target=self._optimization_loop, daemon=True)
            self.thread.start()

    def _log(self, message: str, level: str = "info", to_workspace: bool = False):
        if to_workspace and self.os and hasattr(self.os, 'output_buffer'):
            self.os.output_buffer.append(message)
        
        # Always log to system logger for the 'debug console'
        if level == "success":
            log.info(f"GENETIC SUCCESS: {message}")
        elif level == "warning":
            log.warning(message)
        elif level == "error":
            log.error(message)
        else:
            log.info(message)

    def _get_dynamic_rubric(self) -> Dict[str, Any]:
        """Dynamically scales the rubric to remove hard caps."""
        rubric = copy.deepcopy(self.base_rubric)
        # As generation increases, we increase the max_score of each category
        # and potentially add 'Elite' categories.
        scale_factor = 1.0 + (self.generation // 50) * 0.5 
        for key in rubric:
            rubric[key]["max_score"] *= scale_factor
            
        if self.generation > 100:
            rubric["SOTA Innovation"] = {"expected_content": "Novel connections between disparate scientific fields.", "max_score": 20 * scale_factor}
            
        return rubric

    def _optimization_loop(self):
        self._log("[os.header_accent]🧬 SOTA Genetic Scaffolding Online (ES-Momentum Enabled)[/os.header_accent]", to_workspace=True)
        
        # Baseline
        rubric = self._get_dynamic_rubric()
        test_prompt = random.choice(self.eval_prompts)
        baseline_text = self.vlm.generate(test_prompt, max_new_tokens=50)
        baseline_eval = self.grader.grade_submission(baseline_text, rubric)
        self.best_score = baseline_eval.get("overall_summary", {}).get("total_score", 0.0)
        max_possible = sum(c["max_score"] for c in rubric.values())
        
        self._log(f"[os.status.info]Baseline Genetic Confidence: {self.best_score:.2f}/{max_possible:.1f}[/os.status.info]", to_workspace=True)

        while self._running:
            # Run background optimization at a sustainable cadence
            time.sleep(20) 
            self.generation += 1
            
            try:
                rubric = self._get_dynamic_rubric()
                max_possible = sum(c["max_score"] for c in rubric.values())
                test_prompt = random.choice(self.eval_prompts)
                
                with torch.no_grad():
                    original_weights = self.layer_to_mutate.weight.data.clone()
                    
                    # 1. Sample mutation (using momentum if available)
                    noise = torch.randn_like(original_weights) * self.mutation_scale
                    if self.direction_vector is not None:
                        # Combine random exploration with successful exploitation
                        noise = (1 - self.momentum_beta) * noise + self.momentum_beta * self.direction_vector
                    
                    # 2. Apply mutation
                    self.layer_to_mutate.weight.data.add_(noise)
                
                # 3. Evaluate with diversity
                test_text = self.vlm.generate(test_prompt, max_new_tokens=50, temperature=0.7)
                eval_res = self.grader.grade_submission(test_text, rubric)
                new_score = eval_res.get("overall_summary", {}).get("total_score", 0.0)
                
                # 4. ES Update Logic
                diff = new_score - self.best_score
                
                if diff > 0:
                    # Success: Update best score and momentum
                    self.best_score = new_score
                    if self.direction_vector is None:
                        self.direction_vector = noise
                    else:
                        self.direction_vector = self.momentum_beta * self.direction_vector + (1 - self.momentum_beta) * noise
                    
                    self.mutation_scale *= 1.1 # Increase exploration on success
                    self.convergence_patience = 0
                    self._log(f"[os.status.success]🧬 Generation {self.generation}: Improvement! {new_score:.2f}/{max_possible:.1f} (+{diff:.1f})[/os.status.success]", level="success", to_workspace=True)
                else:
                    # Failure: Revert and decrease scale
                    with torch.no_grad():
                        self.layer_to_mutate.weight.data.copy_(original_weights)
                    
                    self.mutation_scale *= 0.98 # Refine search
                    self.convergence_patience += 1
                    
                    # Log telemetry to system console only
                    self._log(f"Genetic Gen {self.generation}: No improvement ({new_score:.2f})")
                    
                    # If stuck, reset momentum to try new directions
                    if self.convergence_patience > 10:
                        self.direction_vector = None
                        self.mutation_scale = 1e-4
                        self.convergence_patience = 0
                        self._log(f"🧬 Generation {self.generation}: Re-seeding search space due to convergence.", level="warning")

            except Exception as e:
                log.error(f"Genetic mutation error: {e}")
                self._log(f"Genetic Error: {e}", level="error")
                
    def stop(self):
        self._running = False
        if self.thread:
            self.thread.join(timeout=2)
