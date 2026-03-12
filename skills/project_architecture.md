# 🏗️ Project Architecture Skill

## Objective
Enable the agent to understand and navigate its own architectural substrate (SelfResearch OS v3.2).

## Core Systems
1. **The Kernel (`main.py`)**: 
   - Manages the TUI via `prompt_toolkit`.
   - Orchestrates the `_boot_sequence` including auto-calibration.
   - Runs the `run_research_pipeline` which is the primary cognitive loop.
2. **The Substrate (`models/model_wrapper.py` & `config/model_paths.py`)**:
   - High-fidelity model loading and resolution.
   - Cross-OS path safety (Local -> Global -> Scaffold).
   - Environment variable management for third-party isolation.
3. **The Lab (`tools/weight_transfer_tool.py` & others)**:
   - Weight projection, tokenizer offsets, and multi-modal integration.
   - Automated verification via `syntax_checker` and `test_runner`.
4. **The Memory (`memory/persistent_rag.py`)**:
   - Persistent research storage with embedding-based retrieval.
   - Cognitive state persistence (Axioms, Paradoxes, Preferences).

## Cognitive Flow
- **Input** -> **Signal Ingestion** -> **Alien Hypothesis** -> **Dialectical Dissent** -> **Empirical Validation** -> **Commit**.
- Use the `/daemon` for unattended scientific breakthroughs.

## Tool Navigation
- Use `project_indexer` to see the current import map.
- Use `self_researcher` in `self_modify` mode to propose architectural improvements.
