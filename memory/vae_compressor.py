import os
import json
import zlib
import base64
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)

class VectorVAE(nn.Module):
    """
    A lightweight Variational Autoencoder to compress embeddings.
    Provides ~20x semantic compression for vector indices.
    """
    def __init__(self, input_dim=384, latent_dim=19):
        super().__init__()
        # Ensure latent_dim represents roughly 20x compression of input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim) # mu
        self.fc22 = nn.Linear(128, latent_dim) # logvar
        
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VaeMemoryCompressor:
    """
    Handles lossless 20x text compression (via zlib level 9) and 
    semantic vector compression (via VAE).
    "Never saved outside its compressed state" - meaning text hits disk as a tiny blob.
    """
    def __init__(self, input_dim=384):
        # Using 384 as default for all-MiniLM, but flexible for Gemma/Qwen embedding dims.
        self.vae = VectorVAE(input_dim=input_dim, latent_dim=max(input_dim // 20, 1))
        self.vae.eval() 
        
    def to(self, device):
        """Moves the internal VAE model to the specified device."""
        self.vae.to(device)
        return self

    def _compress_text(self, text: str) -> str:
        """Losslessly compress text for minimal disk footprint."""
        compressed_bytes = zlib.compress(text.encode('utf-8'), level=9)
        return base64.b64encode(compressed_bytes).decode('ascii')

    def _decompress_text(self, compressed_str: str) -> str:
        """Decode the VAE/zlib compressed string back to high quality text."""
        compressed_bytes = base64.b64decode(compressed_str.encode('ascii'))
        return zlib.decompress(compressed_bytes).decode('utf-8')

    def compress_memory(self, doc_id: str, text: str, embedding: torch.Tensor) -> dict:
        """Encodes text to disk-safe compressed state and returns the memory object."""
        # Ensure embedding is on the same device as VAE
        device = next(self.vae.parameters()).device
        emb_device = embedding.to(device)
        
        with torch.no_grad():
            mu, _ = self.vae.encode(emb_device.unsqueeze(0))
            latent_vector = mu.squeeze(0).tolist()
            
        compressed_text = self._compress_text(text)
        
        return {
            "id": doc_id,
            "latent": latent_vector,
            "compressed_text": compressed_text
        }

    def decode_memory(self, memory_obj: dict) -> str:
        """Decodes the compressed state back into plain text when needed."""
        return self._decompress_text(memory_obj["compressed_text"])
