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
