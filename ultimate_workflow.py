from __future__ import annotations

"""Ultimate workflow demonstrating all SelfResearch modules with optimized model usage.

This script integrates dataset analysis, training, evaluation,
prompt optimization, simulations, security checks, and
collaboration server interaction in one place using the 270M model.
"""

from typing import Any

import torch
import requests
from models.model_wrapper import DEFAULT_GENERATOR
from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from simulation_lab.experiment_simulator import ExperimentSimulator
from assessment.rubric_grader import RubricGrader
from security.auth_and_ethics import AuthAndEthics
from peer_collab.collaboration_server import CollaborationServer
from data.dataset_loader import load_and_tokenize
from train.trainer import TrainingConfig, train_model
from eval.language_model_evaluator import evaluate_perplexity
from analysis.dataset_analyzer import (
    analyze_tokenized_dataset,
    cluster_dataset_embeddings,
    compute_tsne_embeddings,
)
from analysis.prompt_optimizer import PromptOptimizer
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
from analysis.prompt_bandit_optimizer import PromptBanditOptimizer
from analysis.prompt_bayes_optimizer import BayesianPromptOptimizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize modules
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)
    collab_server = CollaborationServer(device=device)

    # Dataset loading and analysis using centralized generator
    dataset = load_and_tokenize("ag_news", "train[:20]", DEFAULT_GENERATOR)
    stats = analyze_tokenized_dataset(dataset, max_samples=20, include_trigrams=True)
    print("Dataset stats:", stats)

    # Embedding clustering (small subset for speed)
    km, labels = cluster_dataset_embeddings(
        dataset.select(range(10)), DEFAULT_GENERATOR, num_clusters=2, device=device
    )
    print("Cluster labels for first 10 samples:", labels)
    
    # Quick training configuration
    cfg = TrainingConfig(
        model_name=DEFAULT_GENERATOR,
        dataset_name="ag_news",
        train_split="train[:10]",
        eval_split="test[:10]",
        epochs=1,
        batch_size=2,
        output_dir="./demo_model",
    )
    # train_model(cfg)
    
    ppl = evaluate_perplexity(DEFAULT_GENERATOR, "ag_news", split="test[:10]")
    print(f"Perplexity: {ppl:.2f}")

    # Prompt optimization using updated PromptOptimizer
    base_prompt = "Summarize the research article:"
    optim = PromptOptimizer(DEFAULT_GENERATOR)
    print("PromptOptimizer:", optim.optimize(base_prompt, n_variations=2))

    # Research workflow modules
    topic = topic_selector.suggest_topic("AI for healthcare")
    print("Suggested topic:", topic)
    print("Question valid:", topic_selector.validate_question("How can blockchain technology enhance data security in federated learning for medical diagnostics?"))

    # Source evaluation
    evaluation = source_evaluator.evaluate_source("https://example.com")
    print("Source evaluation:", evaluation)

    # Simulation lab
    positions = experiment_simulator.run_physics_simulation(0.0, 5.0, 5, 0.1)
    print("Physics simulation positions:", positions.tolist())

    # Rubric grading
    rubric = {"Quality": {"expected_content": "Detailed study with results", "max_score": 5}}
    grades = rubric_grader.grade_submission("A thorough study with clear results.", rubric)
    print("Grades:", grades)

    # Security & Ethics
    auth_ethics.register_user("demo", "pass", "researcher")
    print("Authenticated:", auth_ethics.authenticate_user("demo", "pass"))
    auth_ethics.flag_ethical_concern("Potential privacy risk in dataset.")
    print("Ethical flags:", auth_ethics.get_ethical_flags())

    print("\nUltimate workflow demonstration concluded.")


if __name__ == "__main__":
    main()
