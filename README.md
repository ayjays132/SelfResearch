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
- `research_workflow/` – topic selection utilities
- `digital_literacy/` – source evaluation and academic search
- `simulation_lab/` – physics/biology simulations and data generation
- `assessment/` – rubric-based grading tools
- `peer_collab/` – collaboration server for shared notes and feedback
- `security/` – user authentication and ethical flagging
- `data/` – helpers for loading and tokenizing datasets

New training loops, datasets or evaluation scripts can be added under new modules, keeping the code organized as described in `AGENTS.md`.

## License
This repository is provided for research and experimentation purposes only.
