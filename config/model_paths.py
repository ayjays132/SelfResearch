import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

# --- PRIORITY RESOLUTION FOR MODELS_DIR ---
GLOBAL_ROOT = Path(os.path.expanduser(os.path.join("~", ".selfresearch")))
REPO_ROOT = Path(__file__).resolve().parents[1]

def get_models_dir() -> str:
    """Resolves the MODELS_DIR in strict priority order.

    Priority is:
    1. Environment override (SELFRESEARCH_MODELS_DIR)
    2. Local repo ./models/ (created if missing)
    3. Global ~/.selfresearch/models/ as fallback
    """

    env_override = os.environ.get("SELFRESEARCH_MODELS_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        env_path.mkdir(parents=True, exist_ok=True)
        return str(env_path)

    local_dir = REPO_ROOT / "models"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        # Use repo-level models directory for reproducibility
        return str(local_dir)
    except PermissionError:
        log.warning("Cannot create local models directory, falling back to global location.")

    global_dir = GLOBAL_ROOT / "models"
    global_dir.mkdir(parents=True, exist_ok=True)
    return str(global_dir)

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
