import os
import json
import logging
from typing import List, Dict, Any
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class TokenizerOffsetManager(BaseTool):
    """
    🔡 TOKENIZER OFFSET MANAGER: Solves token ID collisions in multi-modal models.
    Assigns exclusive vocabulary ranges (offsets) to each modality (Text, Speech, Vision).
    """
    name = "tokenizer_offset_manager"
    description = "Computes cumulative vocabulary offsets for multiple tokenizers to prevent ID collisions. Saves manifest."
    parameters = {
        "type": "object",
        "properties": {
            "tokenizers_info": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "vocab_size": {"type": "integer"}
                    }
                },
                "description": "List of tokenizers with their names and vocab sizes in priority order."
            }
        },
        "required": ["tokenizers_info"]
    }

    def execute(self, tokenizers_info: List[Dict[str, Any]], **kwargs) -> str:
        log.info(f"OffsetManager: Computing offsets for {len(tokenizers_info)} tokenizers.")
        
        offset_manifest = {}
        current_offset = 0
        
        for info in tokenizers_info:
            name = info["name"]
            size = info["vocab_size"]
            
            offset_manifest[name] = {
                "start_id": current_offset,
                "end_id": current_offset + size - 1,
                "vocab_size": size,
                "offset": current_offset
            }
            current_offset += size
            
        total_vocab_size = current_offset
        
        result = {
            "status": "success",
            "total_vocab_size": total_vocab_size,
            "offsets": offset_manifest
        }
        
        # Save to global substrate to ensure cross-workspace consistency
        from config.model_paths import GLOBAL_ROOT
        manifest_path = os.path.join(GLOBAL_ROOT, "tokenizer_offset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(result, f, indent=2)
            
        log.info(f"OffsetManager: Manifest saved to {manifest_path}. Total Vocab: {total_vocab_size}")
        return json.dumps(result, indent=2)
