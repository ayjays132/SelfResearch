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
* `analysis/` – dataset statistics utilities
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
Run it from the command line:

```bash
python3 -m analysis.dataset_analyzer ag_news train distilgpt2 --top-k 3 --trigrams
```

## License
This repository is provided for research and experimentation purposes only.
