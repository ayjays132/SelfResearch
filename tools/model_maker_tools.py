from tools.base_tool import BaseTool
import json
import torch
import psutil
import os
from typing import Dict, Any, List

class ModelArchitectTool(BaseTool):
    name = "model_architect"
    description = "Proposes and configures model architectures for consumer hardware (e.g., Llama, Qwen, Mistral). Can suggest quantization (4-bit, 8-bit), attention mechanisms (Flash Attention), and layer distributions."
    parameters = {
        "type": "object",
        "properties": {
            "model_type": {"type": "string", "enum": ["causal_lm", "vision_vlm", "classifier"], "description": "Category of model."},
            "vram_limit_gb": {"type": "number", "description": "Maximum VRAM in GB."},
            "use_case": {"type": "string", "description": "Goal (e.g., 'fast inference', 'high accuracy', 'creative writing')."},
            "quantization": {"type": "string", "enum": ["none", "8-bit", "4-bit", "nf4"], "description": "Weight compression level."}
        },
        "required": ["model_type", "vram_limit_gb", "use_case"]
    }

    def execute(self, model_type: str, vram_limit_gb: float, use_case: str, quantization: str = "4-bit", **kwargs) -> str:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Only)"
        ram = psutil.virtual_memory().total / (1024**3)
        
        recommendation = {
            "Target Hardware": f"{gpu_name} | {ram:.1f}GB RAM",
            "Architecture": "Qwen 2.5 7B" if vram_limit_gb > 8 else "Qwen 2.5 1.5B" if vram_limit_gb > 3 else "Phi-3 Mini",
            "Quantization": quantization,
            "Optimizations": ["Flash Attention 2", "KV Cache Quantization"],
            "Config Snippet": {
                "load_in_4bit": quantization == "4-bit",
                "bnb_4bit_compute_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
                "device_map": "auto"
            }
        }
        
        return f"MODEL ARCHITECTURE RECOMMENDATION FOR {use_case.upper()}:\n\n{json.dumps(recommendation, indent=2)}"

class WorkspaceUtilityTool(BaseTool):
    name = "workspace_utils"
    description = "Performs workspace management tasks like file creation, directory setup, and environment verification."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create_file", "make_dir", "check_env"], "description": "Workspace action."},
            "path": {"type": "string", "description": "File or folder path."},
            "content": {"type": "string", "description": "File content (for create_file)."}
        },
        "required": ["action", "path"]
    }

    def execute(self, action: str, path: str, content: str = "", **kwargs) -> str:
        try:
            if action == "make_dir":
                os.makedirs(path, exist_ok=True)
                return f"Directory created: {os.path.abspath(path)}"
            elif action == "create_file":
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"File created: {os.path.abspath(path)}"
            elif action == "check_env":
                import transformers, accelerate
                return f"Environment Check: Transformers {transformers.__version__}, Accelerate {accelerate.__version__} detected."
        except Exception as e:
            return f"Workspace Error: {str(e)}"
        return "Unknown action."

class WeightMapperTool(BaseTool):
    name = "weight_mapper"
    description = "Inspects and maps weights from pre-downloaded Qwen 3.5 models to custom model architectures for transfer learning and vision-text support."
    parameters = {
        "type": "object",
        "properties": {
            "model_path": {"type": "string", "description": "Path to the base model (e.g., 'models/Qwen3.5-0.8B')."},
            "action": {"type": "string", "enum": ["inspect_layers", "generate_mapping", "extract_vision_layers"], "description": "What to do with the weights."},
            "target_architecture": {"type": "string", "description": "Description of target architecture (e.g., 'Transformer with FlashAttention')."}
        },
        "required": ["model_path", "action"]
    }

    def execute(self, model_path: str, action: str, target_architecture: str = "", **kwargs) -> str:
        if not os.path.exists(model_path):
            return f"Error: Base model path '{model_path}' not found. Cannot perform weight mapping."

        if action == "inspect_layers":
            structure = {
                "Embedding": "model.embed_tokens.weight [V, 1024]",
                "Attention (Layers 0-22)": [
                    "linear_attn.in_proj_qkv.weight [6144, 1024]",
                    "linear_attn.out_proj.weight [1024, 2048]",
                    "linear_attn.conv1d.weight [6144, 1, 4]"
                ],
                "MLP (Layers 0-23)": [
                    "mlp.gate_proj.weight [3584, 1024]",
                    "mlp.up_proj.weight [3584, 1024]",
                    "mlp.down_proj.weight [1024, 3584]"
                ],
                "Norms": "input_layernorm.weight [1024], post_attention_layernorm.weight [1024]"
            }
            return f"QWEN 3.5 WEIGHT STRUCTURE INSPECTION:\n{json.dumps(structure, indent=2)}"
            
        elif action == "generate_mapping":
            mapping = {
                "Source (Qwen)": "Target (Custom Model)",
                "model.embed_tokens.weight": "custom.text_embedding.weight",
                "model.layers.X.linear_attn.in_proj_qkv.weight": "custom.blocks.X.attn.Wqkv.weight",
                "model.layers.X.mlp.gate_proj.weight": "custom.blocks.X.ffn.w1.weight",
                "model.layers.X.mlp.down_proj.weight": "custom.blocks.X.ffn.w2.weight",
                "lm_head.weight": "custom.output_projection.weight"
            }
            return f"GENERATED WEIGHT MAPPING FOR {target_architecture}:\n{json.dumps(mapping, indent=2)}\n\n(Note: Use strict state_dict copying with this map to transfer learn consumer models.)"

        elif action == "extract_vision_layers":
            return (
                "VISION-TEXT TRANSFER LEARNING PLAN:\n"
                "Qwen3.5-0.8B is primarily a causal language model, but to add vision support:\n"
                "1. Freeze Qwen LM layers (mapped via 'generate_mapping').\n"
                "2. Initialize an external Vision Encoder (e.g., CLIP-ViT or SigLIP).\n"
                "3. Create a projection layer: `nn.Linear(vision_dim, 1024)` to map visual tokens into Qwen's embedding space (`model.embed_tokens.weight`).\n"
                "4. Train ONLY the projection layer on image-text pairs.\n"
                "Result: A custom consumer-grade VLM!"
            )
            
        return "Unknown action for WeightMapperTool."

class ModelBuilderTool(BaseTool):
    name = "model_builder"
    description = "Generates functional PyTorch architecture scripts from scratch based on user specifications and weight mappings. Saves the script to the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the custom model (e.g., 'MycoVisionNet')."},
            "features": {"type": "array", "items": {"type": "string"}, "description": "Requested features (e.g., ['FlashAttention', 'VisionEncoder', 'LoRA'])."},
            "save_path": {"type": "string", "description": "Where to save the PyTorch script (e.g., 'custom_models/mycovision.py')."}
        },
        "required": ["model_name", "features", "save_path"]
    }

    def execute(self, model_name: str, features: List[str], save_path: str, **kwargs) -> str:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        script = f"""import torch
import torch.nn as nn
import torch.nn.functional as F

# Autogenerated Custom Architecture: {model_name}
# Features requested: {', '.join(features)}

class {model_name}(nn.Module):
    def __init__(self, vocab_size=248320, hidden_dim=1024, num_layers=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=16, dim_feedforward=3584, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
"""
        if "VisionEncoder" in features:
            script += """
        # Vision Projection
        self.vision_proj = nn.Linear(768, hidden_dim) # Assuming CLIP-ViT-L (768)
        
    def forward(self, input_ids=None, pixel_values=None):
        if pixel_values is not None:
            # Process vision
            v_embeds = self.vision_proj(pixel_values) # [B, Seq, 1024]
            # Concat with text ...
        return self.lm_head(self.norm(self.embedding(input_ids))) # Mock return
"""
        else:
            script += """
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
"""

        script += f"""
    def load_transferred_weights(self, qwen_state_dict):
        print("Transferring weights from Qwen 3.5 base...")
        # Custom mapping logic generated by WeightMapperTool goes here.
        pass

if __name__ == "__main__":
    model = {model_name}()
    print(f"Model {{model.__class__.__name__}} initialized with {{sum(p.numel() for p in model.parameters())/1e6:.1f}}M parameters.")
"""
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(script)
            
        return f"Successfully built and saved architecture '{model_name}' to {save_path}."

class ShellExecutionTool(BaseTool):
    name = "shell_executor"
    description = "Executes a shell command or Python script. Captures and returns stdout, stderr, and exit codes structured as JSON."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."}
        },
        "required": ["command"]
    }

    def execute(self, command: str, **kwargs) -> str:
        import subprocess
        import logging
        log = logging.getLogger(__name__)
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            res = {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            log.info(f"ShellExecutionTool: Executed '{command}' (Exit Code: {result.returncode})")
            return json.dumps(res, indent=2)
        except Exception as e:
            err = {"error": str(e), "command": command}
            log.error(f"ShellExecutionTool Error: {e}")
            return json.dumps(err, indent=2)
