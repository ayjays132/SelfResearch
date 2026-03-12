from __future__ import annotations

"""Comprehensive workflow demonstrating all modules with optimized model usage."""

import torch
from models.model_wrapper import DEFAULT_GENERATOR
from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from simulation_lab.experiment_simulator import ExperimentSimulator
from assessment.rubric_grader import RubricGrader
from security.auth_and_ethics import AuthAndEthics
from data.dataset_loader import load_and_tokenize
from train.trainer import TrainingConfig, train_model
from eval.language_model_evaluator import evaluate_perplexity
from analysis.dataset_analyzer import analyze_tokenized_dataset
from analysis.prompt_optimizer import PromptOptimizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize modules
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)

    # Dataset loading and analysis
    print("\n=== Dataset Loading & Analysis ===")
    tokenized_ds = load_and_tokenize("ag_news", "train[:50]", DEFAULT_GENERATOR)
    stats = analyze_tokenized_dataset(tokenized_ds, max_samples=50)
    print(f"Dataset stats: {stats}")

    # Training demo (using 270M model)
    print("\n=== Training Demo ===")
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

    # Evaluate perplexity
    ppl = evaluate_perplexity(DEFAULT_GENERATOR, "ag_news", split="test[:10]")
    print(f"Perplexity on small set: {ppl:.2f}")

    # Prompt optimization example
    optimizer = PromptOptimizer(DEFAULT_GENERATOR)
    best_prompt = optimizer.optimize("Summarize the research article:", n_variations=2)
    print(f"Optimized prompt: {best_prompt}")

    # Topic suggestion and question validation
    print("\n=== Research Workflow ===")
    topic = topic_selector.suggest_topic("machine learning for health")
    print(f"Suggested topic: {topic}")
    question = "How can blockchain technology enhance data security in federated learning for medical diagnostics?"
    print(f"Valid question: {topic_selector.validate_question(question)}")

    # Source evaluation
    print("\n=== Source Evaluation ===")
    result = source_evaluator.evaluate_source("https://example.com")
    print(f"Source evaluation result: {result}")

    # Simulation lab
    print("\n=== Simulation Lab ===")
    positions = experiment_simulator.run_physics_simulation(0.0, 5.0, 5, 0.1)
    print(f"Physics simulation positions: {positions.tolist()}")

    # Rubric grading
    print("\n=== Rubric Grading ===")
    rubric = {"Quality": {"expected_content": "Detailed study with methodology", "max_score": 5}}
    grades = rubric_grader.grade_submission("A comprehensive study with clear methodology.", rubric)
    print(f"Grades: {grades}")

    # Authentication and ethics
    print("\n=== Security Module ===")
    auth_ethics.register_user("demo", "pass", "researcher")
    print(f"Authenticated: {auth_ethics.authenticate_user('demo', 'pass')}")
    auth_ethics.flag_ethical_concern("Potential data privacy risk.")
    print(f"Ethical flags: {auth_ethics.get_ethical_flags()}")


if __name__ == "__main__":
    main()
