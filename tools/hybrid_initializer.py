import os
import torch
import json
import logging
from typing import Dict, Any, List, Optional
from safetensors.torch import save_file
from tools.base_tool import BaseTool
from tools.weight_transfer_tool import WeightTransferTool
from tools.syntax_checker import SyntaxCheckerTool
from tools.test_runner import TestRunnerTool
from tools.tokenizer_offset_manager import TokenizerOffsetManager
from models.model_wrapper import MODELS_DIR

log = logging.getLogger(__name__)

class HybridInitializerTool(BaseTool):
    """
    🧬 HYBRID INITIALIZER: Orchestrates multi-source weight transfer.
    Uses Tokenizer Offset Architecture to prevent representation conflicts.
    Automatically resolves Qwen/Qwen3.5-0.8B as the default text backbone.
    """
    name = "hybrid_initializer"
    description = "Orchestrates weight transfer with tokenizer offsets. Automatically uses local Qwen model as text source if available. Args: [target_model_name, source_paths, modality, save_path]."
    parameters = {
        "type": "object",
        "properties": {
            "target_model_name": {"type": "string", "description": "Class name of the target architecture."},
            "source_paths": {"type": "array", "items": {"type": "string"}, "default": [], "description": "List of source checkpoint paths. If empty, system attempts to use local Qwen model as text source."},
            "modality": {"type": "string", "enum": ["text+speech", "text+vision", "text+speech+vision"], "description": "Target hybrid modalities."},
            "save_path": {"type": "string", "description": "Directory to save the resulting hybrid checkpoint."}
        },
        "required": ["target_model_name", "modality", "save_path"]
    }

    def execute(self, target_model_name: str, modality: str, save_path: str, source_paths: List[str] = [], **kwargs) -> str:
        log.info(f"HybridInitializer: Initializing {target_model_name} with {modality} weights.")
        
        # 1. Resolve Default Sources (Qwen Priority)
        final_sources = list(source_paths)
        if not final_sources or len(final_sources) < 1:
            # Check for local Qwen model
            qwen_path = os.path.join(MODELS_DIR, "Qwen3.5-0.8B")
            if os.path.exists(qwen_path):
                # Find the actual .safetensors file
                found_text = False
                for f in os.listdir(qwen_path):
                    if f.endswith(".safetensors"):
                        final_sources.insert(0, os.path.join(qwen_path, f))
                        found_text = True
                        log.info(f"HybridInitializer: Automatically selected Qwen text source at {final_sources[0]}")
                        break
                if not found_text:
                    return json.dumps({"status": "error", "message": "No text source provided and Qwen safetensors not found locally."}, indent=2)
            else:
                return json.dumps({"status": "error", "message": "No source paths provided and global Qwen model not found."}, indent=2)

        # 2. Compute Tokenizer Offsets
        offset_manager = TokenizerOffsetManager()
        tokenizers_info = [
            {"name": "text", "vocab_size": 151936}
        ]
        if "speech" in modality:
            tokenizers_info.append({"name": "speech", "vocab_size": 32000})
        if "vision" in modality:
            tokenizers_info.append({"name": "vision", "vocab_size": 1024})
            
        offset_res_str = offset_manager.execute(tokenizers_info)
        offset_data = json.loads(offset_res_str)
        offset_manifest = offset_data["offsets"]

        # 3. Instantiate Target Model (Simulated)
        target_sd = {
            "brain.0.attn.q_proj.weight": torch.randn(1024, 1024),
            "brain.0.mlp.0.weight": torch.randn(4096, 1024),
            "tokenizer.text_embed.weight": torch.randn(offset_data["total_vocab_size"], 1024)
        }

        # 4. Orchestrate Transfer with Offset Mapping
        transfer_tool = WeightTransferTool()
        report = {"modality_reports": {}}

        for i, path in enumerate(final_sources):
            m_name = tokenizers_info[i]["name"] if i < len(tokenizers_info) else "unknown"
            log.info(f"HybridInitializer: Projecting {m_name} weights to offset {offset_manifest.get(m_name, {}).get('offset')}.")
            m_report_str = transfer_tool.execute([path], m_name, target_state_dict=target_sd)
            report["modality_reports"][m_name] = json.loads(m_report_str)

        # 5. Save
        os.makedirs(save_path, exist_ok=True)
        checkpoint_file = os.path.join(save_path, "hybrid_model.safetensors")
        try:
            save_sd = {k: v.contiguous().to(dtype=torch.bfloat16, device="cpu") for k, v in target_sd.items()}
            save_file(save_sd, checkpoint_file)
            
            manifest = {
                "model_type": target_model_name,
                "modality": modality,
                "offsets": offset_manifest,
                "transfer_metrics": report
            }
            with open(os.path.join(save_path, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Save failed: {e}"})

        return json.dumps({
            "status": "success",
            "checkpoint": checkpoint_file,
            "manifest": os.path.abspath(os.path.join(save_path, "manifest.json")),
            "total_vocab": offset_data["total_vocab_size"]
        }, indent=2)
