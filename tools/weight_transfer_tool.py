import os
import torch
import json
import logging
from typing import Dict, Any, List, Optional, Union
from safetensors.torch import load_file
from tools.base_tool import BaseTool
from models.model_wrapper import MODELS_DIR

log = logging.getLogger(__name__)

class WeightTransferTool(BaseTool):
    """
    🏋️ WEIGHT TRANSFER TOOL: Cross-architecture weight projection.
    Maps source weights to target state dicts using exact, semantic, and shape-projection matching.
    
    KNOWN ARCHITECTURAL CONSTRAINT:
    - Text + Speech = Works ✅
    - Text + Vision = Works ✅
    - Text + Speech + Vision = Representation Conflict ❌
    Root cause: Modality weight distributions compete for shared brain layer dimensions.
    Proposed fix: Modality-specific routing adapters per modality beyond the first.
    """
    name = "weight_transfer_tool"
    description = "Transfers weights from source safetensors to a target state dict. Supports exact, semantic, and shape projection."
    parameters = {
        "type": "object",
        "properties": {
            "source_paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to source safetensors files. Relative to global MODELS_DIR if not absolute."},
            "target_state_dict": {"type": "object", "description": "The target model's state dict."},
            "modality": {"type": "string", "enum": ["text", "speech", "vision", "text+speech", "text+vision", "text+speech+vision"], "description": "Target modalities for transfer."}
        },
        "required": ["source_paths", "modality"]
    }

    def _project_to_shape(self, src_tensor: torch.Tensor, out_shape: torch.Size, target_dtype=torch.bfloat16) -> torch.Tensor:
        """Tile and truncate projection logic."""
        if src_tensor.shape == out_shape:
            return src_tensor.to(dtype=target_dtype, device="cpu")
        
        needed = 1
        for d in out_shape: needed *= d
        
        flat = src_tensor.float().contiguous().view(-1)
        if flat.numel() < needed:
            repeat = (needed + flat.numel() - 1) // flat.numel()
            flat = flat.repeat(repeat)
        
        flat = flat[:needed]
        return flat.view(out_shape).to(dtype=target_dtype, device="cpu")

    def execute(self, source_paths: List[str], modality: str, target_state_dict: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> str:
        log.info(f"WeightTransfer: Initiating {modality} transfer from {len(source_paths)} sources.")
        
        if target_state_dict is None:
            return json.dumps({"error": "target_state_dict must be provided via internal orchestrator."}, indent=2)

        report = {
            "exact_matches": 0,
            "semantic_matches": 0,
            "shape_projections": 0,
            "skipped": 0,
            "modality_routing": {},
            "transferred_keys": []
        }

        is_triple = "speech" in modality and "vision" in modality
        
        for i, path in enumerate(source_paths):
            # Resolve path relative to MODELS_DIR if it's just a folder/filename
            full_path = path
            if not os.path.isabs(path):
                # Try standard relative check first
                if not os.path.exists(path):
                    # Check global substrate
                    full_path = os.path.join(MODELS_DIR, path)
                    # If it's a directory, look for .safetensors inside
                    if os.path.isdir(full_path):
                        for f in os.listdir(full_path):
                            if f.endswith(".safetensors"):
                                full_path = os.path.join(full_path, f)
                                break
            
            if not os.path.exists(full_path):
                log.error(f"Source path not found: {full_path}")
                continue
            
            current_m = "text" if i == 0 else ("speech" if "speech" in modality else "vision")
            if is_triple and i > 0:
                current_m = "speech" if i == 1 else "vision"

            try:
                log.info(f"WeightTransfer: Loading {current_m} source from {full_path}")
                source_state = load_file(full_path)
                for src_key, src_val in source_state.items():
                    matched = False
                    target_key = src_key
                    if src_key in target_state_dict:
                        matched = True
                    else:
                        semantic_map = {"self_attn": "attn", "language_model.layers": "brain", "mlp.linear_fc1": "mlp.0", "mlp.linear_fc2": "mlp.2"}
                        for old, new in semantic_map.items():
                            target_key = target_key.replace(old, new)
                        if target_key in target_state_dict:
                            matched = True
                            report["semantic_matches"] += 1

                    if matched:
                        if is_triple and current_m != "text" and "brain" in target_key:
                            target_state_dict[target_key] = 0.8 * target_state_dict[target_key] + 0.2 * self._project_to_shape(src_val, target_state_dict[target_key].shape)
                            report["modality_routing"][target_key] = f"adapter_injection({current_m})"
                        else:
                            target_state_dict[target_key] = self._project_to_shape(src_val, target_state_dict[target_key].shape)
                            report["modality_routing"][target_key] = f"full_prior({current_m})"
                            if src_val.shape == target_state_dict[target_key].shape:
                                report["exact_matches"] += 1
                            else:
                                report["shape_projections"] += 1
                        report["transferred_keys"].append(src_key)
                    else:
                        report["skipped"] += 1
                        
                del source_state
                torch.cuda.empty_cache()
            except Exception as e:
                log.error(f"Error transferring from {full_path}: {e}")

        return json.dumps(report, indent=2)
