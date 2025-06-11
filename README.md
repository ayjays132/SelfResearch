# SelfResearch Platform

SelfResearch provides a modular environment for experimenting with HuggingFace transformer models using PyTorch. The project bundles several tools for virtual research including topic suggestion, source evaluation, simulation, grading and security management.

## Installation
1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
CUDA is automatically detected. If a GPU is available, PyTorch will use it.

## Running the Example Workflow
Execute `main.py` to launch a demonstration of all modules:
```bash
python3 main.py
```
The script showcases topic selection, source evaluation, running physics and biology simulations, grading a submission and basic authentication.

To start the collaboration server used for peer collaboration, run the following in a separate process:
```bash
python3 peer_collab/collaboration_server.py
```

## Loading Different Models
The modules rely on the HuggingFace `transformers` library. You can edit each component to load any model from the Hub by changing the model names in their initialisation calls. Most small models work well on CPU, while larger models benefit from CUDA acceleration.

## Extending the Pipeline
The project follows a simple structure so new functionality can be added easily:
* `research_workflow/` – topic selection utilities
* `digital_literacy/` – source evaluation and academic search
* `simulation_lab/` – physics/biology simulations and data generation
* `assessment/` – rubric-based grading tools
* `peer_collab/` – collaboration server for shared notes and feedback
* `security/` – user authentication and ethical flagging
* `data/` – helpers for loading and tokenizing datasets
* `analysis/` – dataset statistics utilities and prompt engineering tools
* `models/` – wrappers around HuggingFace models
* `train/` – simple training loops
* `eval/` – evaluation utilities such as perplexity measurement

The `train` module contains utilities for fine-tuning language models with
CUDA support. Training now accepts separate train and evaluation splits and
reports perplexity after each epoch. Run `--help` to see all options:

```bash
python3 -m train.trainer --help
```

The `eval` module includes a script to compute perplexity for a model on a
dataset:

```bash
python3 -m eval.language_model_evaluator gpt2 ag_news --split test
```

## Dataset Caching
`load_and_tokenize` and `load_dataset_splits` accept a `cache_dir` argument. If
provided, tokenized datasets will be saved to disk and loaded on subsequent
calls without reprocessing.

```python
from data.dataset_loader import load_and_tokenize
ds = load_and_tokenize("ag_news", "train[:100]", "distilgpt2", cache_dir="./cache/ag_news_train")
```

`load_and_tokenize` also supports `drop_duplicates=True` to remove repeated
text samples before tokenization.

New training loops, datasets or evaluation scripts can be added under these
modules, keeping the code organized as described in `AGENTS.md`.

## Dataset Analysis
The `analysis` module provides utilities for computing statistics on tokenized
datasets. `analyze_tokenized_dataset` now reports additional metrics such as
token entropy and optional trigram frequencies alongside average length,
vocabulary size and lexical diversity.
The module also includes `cluster_dataset_embeddings` for grouping dataset
samples by semantic similarity using transformer embeddings.
`compute_tsne_embeddings` generates 2D t-SNE coordinates for visualizing
dataset structure.
Run it from the command line:
```bash
python3 -m analysis.dataset_analyzer ag_news train distilgpt2 --top-k 3 --trigrams
```

## Prompt Optimization
The `PromptOptimizer` class generates prompt variations and scores them using perplexity. It returns the best-scoring prompt so you can iteratively refine instructions or dataset prompts.

Example usage:
```python
from analysis.prompt_optimizer import PromptOptimizer
opt = PromptOptimizer("distilgpt2")
print(opt.optimize_prompt("Summarize the research article:"))
```

### Advanced Prompt Optimization
For more precise control, `AdvancedPromptOptimizer` combines perplexity with
semantic similarity of sentence embeddings. This helps retain the intent of the
original prompt while improving clarity.

```python
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
adv = AdvancedPromptOptimizer(
    "distilgpt2", embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"
)
print(adv.optimize_prompt("Summarize the research article:"))
```

### Prompt Augmentation

`PromptAugmenter` generates additional prompt variations to expand training datasets.

```python
from analysis.prompt_augmenter import PromptAugmenter
aug = PromptAugmenter("distilgpt2")
prompts = aug.augment_dataset(["Write an abstract about AI"], n_variations=2)
```

### Evolutionary Prompt Engineering

`PromptEvolver` applies a simple genetic algorithm to refine prompts over several generations.

```python
from analysis.prompt_evolver import PromptEvolver
pe = PromptEvolver("distilgpt2")
print(pe.evolve_prompt("Describe the benefits of renewable energy"))
```

### Bandit-Based Prompt Optimization

`PromptBanditOptimizer` employs an epsilon-greedy multi-armed bandit strategy
to iteratively explore and exploit prompt variations based on a custom reward
function.

```python
from analysis.prompt_bandit_optimizer import PromptBanditOptimizer

# Reward function prefers longer prompts
bandit = PromptBanditOptimizer("distilgpt2", reward_fn=len, epsilon=0.1)
best = bandit.optimize_prompt("Write a summary")
print(best)
```



## License
This repository is provided for research and experimentation purposes only.
