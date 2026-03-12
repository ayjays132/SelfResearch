# 🏋️ Weight Transfer Lab Skill

## Objective
Enable the agent to perform cross-architecture weight projection to initialize hybrid models (e.g., Phillux) using pre-trained weights from disparate sources (Qwen, MobileNet, SDXL).

## Protocol
1. **Source Loading**: Load source weights from `.safetensors` files.
2. **Key Mapping**:
   - **Exact**: 1:1 key name matching.
   - **Semantic**: Fuzzy matching for common architectural name variations (e.g., `self_attn` -> `attn`).
   - **Shape Projection**: Use the `Tile + Truncate` algorithm to project source tensors into the target shape when dimensions mismatch.
3. **Modality Constraint**: 
   - Supports **Text + Speech** OR **Text + Vision**.
   - **FATAL CONSTRAINT**: Never attempt all three simultaneously. Representation conflict at shared layers leads to catastrophic intelligence loss.
4. **Verification**:
   - Save the hybrid checkpoint.
   - Run the **Coherence Evaluator** to measure intelligence retention (0-100 score).
   - A score > 70 confirms a valid projection.

## Tools
- `weight_transfer_tool`: The mapping engine.
- `hybrid_initializer`: The orchestrator.
- `coherence_evaluator`: The verification gate.
