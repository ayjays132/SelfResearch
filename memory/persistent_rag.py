import os
import json
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from models.model_wrapper import ModelRegistry, DEFAULT_EMBEDDER
from memory.vae_compressor import VaeMemoryCompressor

log = logging.getLogger(__name__)

class PersistentRAG:
    """
    A persistent vector store that uses VAE for heavy memory compression (20x)
    and FAISS for sub-millisecond similarity search at scale.
    
    Architecture:
    1. VAE: Compresses high-dim embeddings into a small latent space.
    2. zlib: Losslessly compresses text on disk.
    3. FAISS: Indexes the latent space for high-speed retrieval.
    """
    def __init__(self, store_path="memory_store/rag_store.json", device=None):
        self.store_path = store_path
        self.device = ModelRegistry.get_device(device)
        self.embedder = ModelRegistry.get_embedder(DEFAULT_EMBEDDER, self.device)
        
        # We need to dynamically set VAE input_dim based on the embedder
        test_emb = self.embedder.encode("test", convert_to_tensor=True)
        self.compressor = VaeMemoryCompressor(input_dim=test_emb.shape[-1]).to(self.device)
        
        # Determine latent dimension for FAISS index
        self.latent_dim = self.compressor.vae.fc21.out_features
        
        # Initialize the FAISS index (Inner Product for Cosine Similarity on normalized vectors)
        if faiss:
            self.index = faiss.IndexFlatIP(self.latent_dim)
            log.info(f"FAISS index (IP) initialized for latent dimension {self.latent_dim}.")
        else:
            self.index = None
            log.warning("FAISS not found. Falling back to sequential Torch-based similarity.")
        
        self.memory_store: List[dict] = []
        self.load_store()

    def load_store(self):
        """Loads the JSON store and populates the in-memory FAISS index."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.memory_store = json.load(f)
                
                if self.memory_store and self.index:
                    # Extract and normalize latents for the FAISS index
                    latents = np.array([mem["latent"] for mem in self.memory_store]).astype('float32')
                    # Normalize for Cosine Similarity (Inner Product on L2-normalized vectors)
                    faiss.normalize_L2(latents)
                    self.index.add(latents)
                    
                log.info(f"Loaded {len(self.memory_store)} compressed memories. FAISS index populated.")
            except Exception as e:
                log.error(f"Failed to load RAG store: {e}")
                self.memory_store = []

    def save_store(self):
        """Saves the JSON representation to disk (remains backward compatible)."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_store, f)
        log.info(f"Saved {len(self.memory_store)} memories to '{self.store_path}'.")

    def add_document(self, doc_id: str, text: str):
        """Encodes, compresses, and adds a document to both disk and the FAISS index."""
        # Check if already exists
        if any(mem["id"] == doc_id for mem in self.memory_store):
            return

        embedding = self.embedder.encode(text, convert_to_tensor=True).to(self.device)
        compressed_memory = self.compressor.compress_memory(doc_id, text, embedding)
        
        # Add to FAISS index immediately
        if self.index:
            latent = np.array([compressed_memory["latent"]]).astype('float32')
            faiss.normalize_L2(latent)
            self.index.add(latent)
            
        self.memory_store.append(compressed_memory)
        self.save_store()
        log.info(f"Document '{doc_id}' added. FAISS index updated incrementally.")

    def query(self, query_text: str, top_k: int = 2) -> List[Dict[str, str]]:
        """
        High-speed query using the FAISS index.
        Returns the most relevant compressed documents after VAE-decoding.
        """
        if not self.memory_store:
            return []

        # 1. Encode query to latent space
        query_emb = self.embedder.encode(query_text, convert_to_tensor=True).to(self.device)
        with torch.no_grad():
            query_latent, _ = self.compressor.vae.encode(query_emb.unsqueeze(0))
            query_latent_np = query_latent.cpu().numpy().astype('float32')

        # 2. Perform Similarity Search
        if self.index:
            faiss.normalize_L2(query_latent_np)
            # D = distances (Inner Product), I = indices
            top_k = min(top_k, self.index.ntotal)
            D, I = self.index.search(query_latent_np, top_k)
            
            top_indices = I[0]
            scores = D[0]
        else:
            # Fallback to pure Torch similarity (Linear scan)
            log.debug("Falling back to Torch-based similarity search.")
            store_latents = torch.tensor([mem["latent"] for mem in self.memory_store]).to(self.device)
            similarities = F.cosine_similarity(query_latent, store_latents)
            top_k = min(top_k, len(self.memory_store))
            scores, top_indices = torch.topk(similarities, top_k)
            top_indices = top_indices.tolist()
            scores = scores.tolist()

        # 3. Process and Decode Results
        results = []
        for idx, score in zip(top_indices, scores):
            if idx == -1: continue # FAISS empty result indicator
            mem = self.memory_store[int(idx)]
            # VAE Decode text back from its zlib-compressed state
            decompressed_text = self.compressor.decode_memory(mem)
            results.append({
                "id": mem["id"],
                "text": decompressed_text,
                "score": float(score)
            })
            
        return results
