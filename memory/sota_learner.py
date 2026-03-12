import json
import os
import zlib
import base64
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

log = logging.getLogger(__name__)

class SOTALearner:
    """
    State-Of-The-Art (SOTA) Learner & Cross-Project Innovation Substrate.
    Compresses 'Alien-like' architectural innovations using lossless text compression
    combined with semantic embeddings (RAG) for cross-project intelligence transfer.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.global_store_dir = "memory_store/sota_latent_cache"
        os.makedirs(self.global_store_dir, exist_ok=True)
        self.store_file = os.path.join(self.global_store_dir, "global_sota_registry.json")
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()
        
        # Initialize semantic substrate for cross-modal retrieval
        from models.model_wrapper import ModelRegistry, DEFAULT_EMBEDDER
        self.embedder = ModelRegistry.get_embedder(DEFAULT_EMBEDDER, device=self.device)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Failed to load SOTA registry: {e}")
        return {}

    def _save_registry(self):
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=4)

    def _compress_payload(self, text: str) -> str:
        """Lossless high-density compression for raw code and theory."""
        compressed = zlib.compress(text.encode('utf-8'), level=9)
        return base64.b64encode(compressed).decode('utf-8')

    def _decompress_payload(self, encoded_str: str) -> str:
        try:
            decoded = base64.b64decode(encoded_str.encode('utf-8'))
            return zlib.decompress(decoded).decode('utf-8')
        except Exception as e:
            return f"Error decompressing SOTA payload: {e}"

    def register_innovation(self, concept_name: str, domain: str, technical_payload: str, empirical_score: float, force_update: bool = False) -> str:
        """
        Validates and compresses a SOTA concept. 
        If concept exists, compares scores to ensure the registry only keeps the most 'Advanced' version.
        """
        concept_key = concept_name.strip()
        
        if concept_key in self.registry and not force_update:
            old_score = self.registry[concept_key].get("validation_score", 0.0)
            if empirical_score <= old_score:
                return f"Innovation '{concept_key}' already exists with a superior or equal score ({old_score}). Skipping update."
            else:
                log.info(f"Evolving SOTA concept '{concept_key}': Score {old_score} -> {empirical_score}")

        compressed_data = self._compress_payload(technical_payload)
        
        # Generate semantic fingerprint (The RAG component)
        summary = f"SOTA {domain} Concept: {concept_key}. Payload Focus: {technical_payload[:200]}"
        
        # SILENCE ENTIRE EMBEDDING CALL
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            embedding = self.embedder.encode(summary).tolist()
        
        self.registry[concept_key] = {
            "domain": domain,
            "validation_score": empirical_score,
            "payload": compressed_data,
            "latent_vector": embedding,
            "timestamp": datetime.now().isoformat(),
            "version": self.registry.get(concept_key, {}).get("version", 0) + 1
        }
        self._save_registry()
        return f"Successfully archived SOTA '{concept_key}' (Version {self.registry[concept_key]['version']}) to Global Substrate."

    def find_similar_innovation(self, query: str, threshold: float = 0.5) -> List[str]:
        """Uses RAG to find SOTA breakthroughs relevant to a new query."""
        if not self.registry: return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            query_vec = self.embedder.encode(query).reshape(1, -1)
        
        results = []
        for name, data in self.registry.items():
            latent_vec = np.array(data["latent_vector"]).reshape(1, -1)
            sim = float(cosine_similarity(query_vec, latent_vec)[0][0])
            if sim >= threshold:
                results.append(name)
        return results

    def retrieve_innovation(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a SOTA concept and returns its raw data for high-fidelity rendering.
        """
        if concept_name in self.registry:
            data = self.registry[concept_name]
            decompressed = self._decompress_payload(data["payload"])
            
            return {
                "name": concept_name,
                "domain": data["domain"],
                "content": decompressed,
                "score": data["validation_score"],
                "version": data.get("version", 1)
            }
        return None

    def list_innovations(self, domain_filter: Optional[str] = None) -> str:
        if not self.registry:
            return "No SOTA innovations currently registered in the global cache."
        
        results = []
        for name, data in self.registry.items():
            if domain_filter and domain_filter.lower() not in data["domain"].lower():
                continue
            results.append(f"- {name} [{data['domain']}] (Validation: {data['validation_score']})")
            
        if not results:
            return f"No innovations found for domain: {domain_filter}"
            
        return "GLOBAL SOTA REGISTRY:\n" + "\n".join(results)
