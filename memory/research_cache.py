import json
import os
import hashlib
from typing import Dict, Optional

class NeuralLatentCache:
    """
    Persistent cache for research syntheses and latent insights.
    Prevents redundant LLM calls and increases protocol speed.
    """
    def __init__(self, project_name: str):
        self.cache_dir = "memory_store/research_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.project_name = project_name
        self.cache_file = os.path.join(self.cache_dir, f"{self._sanitize(project_name)}_cache.json")
        self.data: Dict[str, str] = self._load()

    def _sanitize(self, name: str) -> str:
        return "".join([c if c.isalnum() else "_" for c in name])

    def _load(self) -> Dict[str, str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get(self, topic: str) -> Optional[str]:
        # Hash the topic to use as key
        topic_hash = hashlib.md5(topic.lower().strip().encode()).hexdigest()
        return self.data.get(topic_hash)

    def set(self, topic: str, synthesis: str):
        topic_hash = hashlib.md5(topic.lower().strip().encode()).hexdigest()
        self.data[topic_hash] = synthesis
        self._save()

    def _save(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def clear(self):
        self.data = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
