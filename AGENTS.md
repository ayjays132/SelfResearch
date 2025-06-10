<<<<<<< HEAD
# AGENTS Guidelines for SelfResearch

This repository provides a prototype pipeline for training and experimenting with
HuggingFace language models using PyTorch with CUDA acceleration.
The baseline example focuses on the `gpt2-medium` causal language model but the
pipeline should remain flexible enough to accommodate most transformer models
available through HuggingFace.

## Objectives
- Build a clean, modular training pipeline in Python leveraging the
  HuggingFace `transformers` and `datasets` libraries.
- Support GPU acceleration with CUDA via PyTorch. Ensure that CPU execution also
  works for environments without a GPU.
- Encourage exploration and fine-tuning of different models. GPT‑2 medium is the
  default, but any model from the HuggingFace hub should be usable with minimal
  configuration changes.
- Provide utilities for experiment tracking, evaluation, and dataset management
  to facilitate reproducible research.
- Foster the agent's ability to research and extend the codebase by exploring
  new features, training schedules, or model architectures.

## Coding Conventions
- Use Python 3.10+ with type hints.
- Follow PEP 8 style guidelines. Use descriptive variable names and add
  explanatory comments for complex logic.
- Organize code into clear modules: `data/` for dataset loading and tokenization,
  `models/` for model wrappers, `train/` for training loops, and `eval/` for
  evaluation scripts.
- When possible, provide small runnable examples or unit tests to validate core
  functionality. Use `pytest` for tests.

## Workflow Guidelines
1. **Environment Setup**
   - Depend on `torch`, `transformers`, and `datasets`. Provide a `requirements.txt`.
   - Enable optional CUDA usage by detecting `torch.cuda.is_available()` and
     setting device accordingly.
2. **Dataset Preparation**
   - Tokenize datasets using the model tokenizer (e.g., GPT‑2 tokenizer) with
     attention to efficient batching and padding.
   - Allow loading custom datasets from local files or the HuggingFace Hub.
3. **Training Loop**
   - Implement a standard training loop with gradient accumulation support and
     periodic evaluation on a validation set.
   - Include options for learning rate schedules, mixed precision training, and
     checkpoint saving.
4. **Evaluation and Research**
   - Provide metrics such as perplexity for language models, and allow easy
     extension for custom metrics.
   - Encourage experimentation with prompts or downstream tasks to enable the
     model to produce novel insights and discoveries.
5. **Documentation**
   - Maintain clear README files and inline documentation. Describe how to run
     training, how to load different models, and how to extend the pipeline.

## Notes for Codex Agents
- Respect the modular structure when adding or modifying files.
- Keep code and documentation concise but thorough.
- If you introduce new dependencies, update `requirements.txt`.
- Before committing, run `pytest` if tests are provided. If tests fail or cannot
  be run, note it in the PR message.
=======
# AI Agents in the Virtual Research Education Platform

This document outlines the various AI agents integrated into the Virtual Research Education Platform, detailing their roles and functionalities.

## 1. Topic Selector Agent

**Module:** `research_workflow/topic_selector.py`

**Role:** Assists users in identifying and validating research topics and questions. It leverages a pre-trained language model to suggest novel and specific research topics based on a given research area. It also includes a validation mechanism to assess the clarity, focus, and feasibility of user-defined research questions.

**Key Features:**
- Generates specific research topics using a generative LLM.
- Validates research questions for coherence and relevance.
- Integrates PyTorch for underlying NLP computations.

## 2. Source Evaluation Agent

**Module:** `digital_literacy/source_evaluator.py`

**Role:** Evaluates the credibility, bias, and relevance of academic sources (URLs, simulated PDFs/metadata). It integrates with academic search APIs (e.g., arXiv, Semantic Scholar) to fetch real-world academic data and employs NLP techniques for in-depth analysis.

**Key Features:**
- Fetches content from URLs and simulates content for other source types.
- Assesses source credibility using zero-shot classification.
- Detects potential bias through sentiment analysis and political bias classification.
- Summarizes source content for quick understanding.
- Searches academic databases for relevant papers.

## 3. Experiment Simulation Agent

**Module:** `simulation_lab/experiment_simulator.py`

**Role:** Provides a virtual environment for running and visualizing scientific simulations. It supports various models, including physics simulations (e.g., projectile motion) and biological simulations (e.g., Lotka-Volterra predator-prey models). It also generates synthetic datasets for experimental purposes.

**Key Features:**
- Runs physics and biological simulations with configurable parameters.
- Generates synthetic datasets with controllable noise levels.
- Visualizes simulation results and generated data using Matplotlib.
- Utilizes PyTorch for accelerated numerical computations.

## 4. Rubric Grading Agent

**Module:** `assessment/rubric_grader.py`

**Role:** Automates the grading of research proposals and reports based on predefined rubrics. It uses advanced NLP techniques, including transformer models and sentence embeddings, to compare submission content against expected criteria and provide detailed, empathetic feedback.

**Key Features:**
- Calculates similarity between submission text and rubric criteria using sentence embeddings.
- Assigns scores based on content similarity.
- Generates constructive and empathetic feedback using a text generation model.
- Leverages PyTorch for efficient NLP model inference.

## 5. Authentication and Ethics Agent

**Module:** `security/auth_and_ethics.py`

**Role:** Manages user authentication, access control, and ethical review processes within the platform. It ensures secure user access, defines role-based permissions, and provides mechanisms for flagging and reviewing ethical concerns related to research activities or data handling.

**Key Features:**
- Secure user registration and authentication with password hashing.
- Role-based access control for different platform functionalities.
- System for flagging and reviewing ethical concerns with detailed tracking.
- Ensures data privacy and responsible research practices.

These agents collectively form the backbone of the Virtual Research Education Platform, providing intelligent assistance and automation across various stages of the research and learning workflow.


>>>>>>> 714aa31 (Initial full project import)
