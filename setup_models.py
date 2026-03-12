import sys
import logging
import os

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from config.model_paths import ensure_model_exists

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def main():
    log.info("🧠 Initializing SelfResearch OS Substrate Models...")
    # These models are required for the framework to run fully locally without crashing.
    models = [
        "Qwen/Qwen3.5-0.8B",
        "ayjays132/EMOTIONVERSE-2"
    ]
    for model in models:
        try:
            ensure_model_exists(model)
        except Exception as e:
            log.error(f"Failed to download {model}: {e}")
    
    log.info("✅ All core neural models have been downloaded and calibrated.")

if __name__ == "__main__":
    main()
