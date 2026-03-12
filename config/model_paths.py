import os
import logging
from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

# --- PRIORITY RESOLUTION FOR MODELS_DIR ---
GLOBAL_ROOT = os.path.expanduser(os.path.join("~", ".selfresearch"))

def get_models_dir() -> str:
    """
    Resolves the MODELS_DIR in strict priority order:
    1. Local: ./models/ relative to CWD
    2. Global: ~/.selfresearch/models/
    3. Auto-Scaffold: Create ~/.selfresearch/models/
    """
    local_dir = os.path.join(os.getcwd(), "models")
    global_dir = os.path.join(GLOBAL_ROOT, "models")

    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        return local_dir
    
    if os.path.exists(global_dir) and os.path.isdir(global_dir):
        return global_dir

    # Fallback: Auto-create global
    os.makedirs(global_dir, exist_ok=True)
    return global_dir

MODELS_DIR = get_models_dir()

def resolve_model_path(repo_id: str) -> str:
    """
    Bulletproof resolution of a model path within the solved MODELS_DIR.
    Handles standard repos like 'Qwen/Qwen3.5-0.8B' -> 'Qwen3.5-0.8B' folder mapping.
    """
    mapping = {
        "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B",
        "ayjays132/EMOTIONVERSE-2": "EMOTIONVERSE-2",
        "google/embeddinggemma-300m": "EmbeddingGemma-300m"
    }
    folder_name = mapping.get(repo_id, repo_id.replace("/", "--"))
    return os.path.join(MODELS_DIR, folder_name)

def ensure_model_exists(repo_id: str):
    """Checks for model weights and downloads if missing."""
    path = resolve_model_path(repo_id)
    has_weights = False
    if os.path.exists(path):
        has_weights = any(os.path.exists(os.path.join(path, f)) for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors-00001-of-00001.safetensors"])
    
    if not has_weights:
        log.info(f"Substrate missing: '{repo_id}'. Downloading to {path}...")
        snapshot_download(repo_id=repo_id, local_dir=path, local_dir_use_symlinks=False)
        log.info(f"Model '{repo_id}' calibrated successfully.")
    else:
        log.info(f"Model '{repo_id}' verified at {path}.")
    return path
