import json
import os
from typing import Any, Dict

SETTINGS_FILE = "settings.json"
GLOBAL_ROOT = os.path.expanduser(os.path.join("~", ".selfresearch"))
GLOBAL_SETTINGS = os.path.join(GLOBAL_ROOT, SETTINGS_FILE)

DEFAULT_SETTINGS = {
    "hardware_acceleration": "auto",
    "enable_genetic_mutation": True,
    "visual_browser_headless": True,
    "max_new_tokens": 800,
    "theme": "dark",
    "clear_cuda_cache_on_mode_switch": True,
    "yolo_mode": False,
    "active_model_provider": "local", # local, ollama, openai, gemini
    "active_model_name": "Qwen/Qwen3.5-0.8B",
    "ollama_base_url": "http://localhost:11434",
    "openai_api_key": "",
    "gemini_api_key": ""
}

class SettingsManager:
    @staticmethod
    def _get_path() -> str:
        # Prioritize local settings in CWD for project isolation
        if os.path.exists(SETTINGS_FILE):
            return SETTINGS_FILE
        # Fallback to global user settings
        if os.path.exists(GLOBAL_SETTINGS):
            return GLOBAL_SETTINGS
        # Default to local if neither exist (creation point)
        return SETTINGS_FILE

    @staticmethod
    def load() -> Dict[str, Any]:
        target = SettingsManager._get_path()
        if not os.path.exists(target):
            # If no settings found anywhere, create a global base for the user
            os.makedirs(GLOBAL_ROOT, exist_ok=True)
            SettingsManager.save(DEFAULT_SETTINGS, global_scope=True)
            return DEFAULT_SETTINGS.copy()
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(data)
                return merged
        except Exception:
            return DEFAULT_SETTINGS.copy()

    @staticmethod
    def save(settings: Dict[str, Any], global_scope: bool = False):
        path = GLOBAL_SETTINGS if global_scope else SETTINGS_FILE
        # Ensure directory exists if saving globally
        if global_scope:
            os.makedirs(GLOBAL_ROOT, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
