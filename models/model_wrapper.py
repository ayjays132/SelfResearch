from __future__ import annotations

import os
import sys
import logging
import json
import re
import requests
from typing import Optional, Any, Dict, Union, List

# --- UNIFIED PATH RESOLUTION ---
from config.model_paths import MODELS_DIR, resolve_model_path, ensure_model_exists

# Set environment variables to force all third-party libraries into our substrate
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(MODELS_DIR, "datasets")
# --------------------------------

log = logging.getLogger(__name__)

import transformers
transformers.logging.set_verbosity_error()

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoModelForImageTextToText,
    AutoTokenizer, 
    AutoProcessor,
    GenerationConfig,
    pipeline
)
from huggingface_hub import snapshot_download

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class ModelRegistry:
    """
    A singleton-like registry to manage model instances.
    Utilizes config/model_paths.py for cross-OS path safety.
    """
    _instance = None
    _models: Dict[str, Any] = {}
    _processors: Dict[str, Any] = {}
    _pipelines: Dict[str, Any] = {}
    _embedders: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_hardware_config(cls) -> Dict[str, Any]:
        config = {"device": "cpu", "dtype": torch.float32}
        if torch.cuda.is_available():
            config["device"] = "cuda"
            config["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config["device"] = "mps"
            config["dtype"] = torch.float16
        return config

    @classmethod
    def get_device(cls, device_str: Optional[str] = None) -> torch.device:
        if device_str: return torch.device(device_str)
        return torch.device(cls.get_hardware_config()["device"])

    @classmethod
    def get_local_path(cls, repo_id: str) -> str:
        return resolve_model_path(repo_id)

    @classmethod
    def ensure_model_locally(cls, repo_id: str) -> str:
        return ensure_model_exists(repo_id)

    @classmethod
    def get_model_and_processor(cls, model_name: str, device: Optional[str] = None, model_type: str = "vlm"):
        hw_config = cls.get_hardware_config()
        device_obj = torch.device(device) if device else torch.device(hw_config["device"])
        local_path = cls.ensure_model_locally(model_name)
        
        cache_key = f"{model_name}_{device_obj.type}"
        
        if cache_key not in cls._models:
            log.info(f"Loading model from {local_path} on {device_obj}...")
            
            load_args = {
                "dtype": hw_config["dtype"],
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            if device_obj.type != "cpu":
                load_args["device_map"] = "auto"

            try:
                if "EMOTIONVERSE" in model_name or model_type == "classification":
                    model = AutoModelForSequenceClassification.from_pretrained(local_path, **load_args)
                    processor = AutoTokenizer.from_pretrained(local_path)
                elif model_type == "vlm" or "Qwen3.5" in model_name:
                    model = AutoModelForImageTextToText.from_pretrained(local_path, **load_args)
                    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
                else:
                    model = AutoModelForCausalLM.from_pretrained(local_path, **load_args)
                    processor = AutoTokenizer.from_pretrained(local_path)
                
                if "device_map" not in load_args or load_args["device_map"] is None:
                    model = model.to(device_obj)

            except Exception as e:
                log.error(f"Local load failed for {local_path}: {e}. Retrying with CPU fallback.")
                if "EMOTIONVERSE" in model_name or model_type == "classification":
                    model = AutoModelForSequenceClassification.from_pretrained(local_path).to("cpu")
                else:
                    model = AutoModelForCausalLM.from_pretrained(local_path).to("cpu")
                processor = AutoTokenizer.from_pretrained(local_path)

            model.eval()
            cls._models[cache_key] = model
            cls._processors[cache_key] = processor
            log.info(f"Model and processor cached.")
            
        return cls._models[cache_key], cls._processors[cache_key]

    @classmethod
    def get_pipeline(cls, task: str, model_name: str, device: Optional[str] = None, **kwargs):
        device_obj = cls.get_device(device)
        device_id = 0 if device_obj.type == 'cuda' else -1
        local_path = cls.ensure_model_locally(model_name)
        
        cache_key = f"pipe_{task}_{model_name}_{device_id}"
        
        if cache_key not in cls._pipelines:
            log.info(f"Initializing pipeline '{task}' with '{model_name}' on {device_obj}...")
            hw = cls.get_hardware_config()
            cls._pipelines[cache_key] = pipeline(
                task, model=local_path, device=device_id, torch_dtype=hw["dtype"], **kwargs
            )
            log.info(f"Successfully cached pipeline.")
            
        return cls._pipelines[cache_key]

    @classmethod
    def get_embedder(cls, model_name: str, device: Optional[str] = None):
        if SentenceTransformer is None: raise ImportError("sentence-transformers missing.")
        device_obj = torch.device(device) if device else torch.device(cls.get_hardware_config()["device"])
        local_path = cls.ensure_model_locally(model_name)
        
        cache_key = f"embed_{model_name}_{device_obj.type}"
        if cache_key not in cls._embedders:
            model = SentenceTransformer(local_path, device=str(device_obj))
            model.show_progress_bar = False # Silence tqdm batches
            cls._embedders[cache_key] = model
        return cls._embedders[cache_key]

class LanguageModelWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", device: Optional[str] = None, provider: str = "local") -> None:
        self.model_name = model_name
        self.device_str = device
        self.provider = provider
        self.enable_peer_review = False
        
        # Only load local weights if provider is 'local' or it's a classification model (EmotionVerse)
        if provider == "local" or "EMOTIONVERSE" in model_name:
            m_type = "classification" if "EMOTIONVERSE" in model_name else "vlm"
            self.model, self.processor = ModelRegistry.get_model_and_processor(model_name, device, model_type=m_type)
            self.device = self.model.device
        else:
            self.model = None
            self.processor = None
            self.device = "cpu"

    def generate(self, prompt: str, image_url: Optional[str] = None, image: Optional[Any] = None, use_tools: bool = False, tool_manager: Optional[Any] = None, system_prompt: Optional[str] = None, n_variations: int = 1, **kwargs) -> Union[str, List[str]]:
        # Build the full input messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        tool_instr = ""
        if use_tools and tool_manager:
            tool_instr = tool_manager.get_tools_prompt()
        
        user_content = []
        if image_url: user_content.append({"type": "image", "url": image_url})
        if image: user_content.append({"type": "image", "image": image})
        
        full_user_text = f"{tool_instr}\n\nTask: {prompt}" if tool_instr else prompt
        user_content.append({"type": "text", "text": full_user_text})
        messages.append({"role": "user", "content": user_content})

        max_loops = 5
        current_loop = 0
        
        while current_loop < max_loops:
            current_loop += 1
            
            # Dispatch based on provider
            if self.provider == "local" or "EMOTIONVERSE" in self.model_name:
                decoded = self._generate_local(messages, n_variations=n_variations, **kwargs)
            elif self.provider == "ollama":
                decoded = self._generate_ollama(messages, **kwargs)
            elif self.provider == "openai":
                decoded = self._generate_openai(messages, **kwargs)
            elif self.provider == "gemini":
                if self.model_name == "auto":
                    decoded = self._generate_gemini_auto(messages, **kwargs)
                else:
                    decoded = self._generate_gemini(messages, self.model_name, **kwargs)
            else:
                return f"Error: Provider '{self.provider}' not supported."

            # Clean output
            if isinstance(decoded, list):
                decoded = [self._clean_output(d) for d in decoded]
            else:
                decoded = self._clean_output(decoded)
            
            if not use_tools:
                return decoded
                
            # Use the provided tool manager or fallback (Tool usage only supports single string)
            tm = tool_manager if tool_manager else ToolManager()
            called, result = tm.parse_and_execute(decoded if isinstance(decoded, str) else decoded[0])
            
            if called:
                # Add history for next iteration
                messages.append({"role": "assistant", "content": [{"type": "text", "text": decoded if isinstance(decoded, str) else decoded[0]}]})
                
                # --- PARALLEL PEER REVIEW INJECTION ---
                reviewer_insight = ""
                if getattr(self, "enable_peer_review", False):
                    from peer_collab.active_reviewer import get_active_reviewer
                    reviewer = get_active_reviewer(self.model_name)
                    self.enable_peer_review = False
                    reviewer_insight = reviewer.review_action(decoded if isinstance(decoded, str) else decoded[0], result)
                    self.enable_peer_review = True
                    
                final_result = result
                if reviewer_insight:
                    final_result += f"\n\n[Parallel Peer Reviewer Insight]: {reviewer_insight}"
                    
                messages.append({"role": "user", "content": [{"type": "text", "text": final_result}]})
            else:
                return decoded
                
        return "Loop limit reached."

    def _generate_local(self, messages: List[Dict], n_variations: int = 1, **kwargs) -> Union[str, List[str]]:
        if hasattr(self.processor, "apply_chat_template"):
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.device)
        else:
            text_input = messages[-1]["content"][-1]["text"]
            inputs = self.processor(text_input, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 400),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "num_return_sequences": n_variations
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        input_len = inputs["input_ids"].shape[-1]
        if n_variations > 1:
            return [self.processor.decode(out[input_len:], skip_special_tokens=True) for out in outputs]
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _generate_gemini_auto(self, messages: List[Dict], **kwargs) -> str:
        """Gracefully degrades through Gemini models: 3.1 Pro -> 3.1 Flash -> 2.5 Flash."""
        tier_list = [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash"
        ]
        
        errors = []
        for model in tier_list:
            res = self._generate_gemini(messages, model, **kwargs)
            if not res.startswith("Gemini Error:"):
                return res
            
            # Catch specific critical errors to stop early
            if "billing" in res.lower() or "quota" in res.lower() or "429" in res:
                errors.append(f"[{model}] {res}")
                break # Rate limits usually apply across the project
            
            errors.append(f"[{model}] Unavailable.")

        error_msg = "\n".join(errors)
        return f"Gemini Auto-Mode Failed Cycle:\n{error_msg}\n\n[RECOMENDATION]: Check API key, billing status, or rate limits. Press Enter to retry."

    def _generate_gemini(self, messages: List[Dict], model_override: Optional[str] = None, **kwargs) -> str:
        from settings_manager import SettingsManager
        settings = SettingsManager.load()
        key = settings.get("gemini_api_key")
        if not key: return "Error: Gemini API Key missing in settings."
        
        model_name = model_override if model_override else self.model_name
        if "gemini" not in model_name or "1.5" in model_name:
            model_name = "gemini-2.5-flash"
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        
        contents = []
        for m in messages:
            parts = []
            for c in m["content"]:
                if c["type"] == "text": parts.append({"text": c["text"]})
                if c["type"] == "image":
                    import base64
                    from io import BytesIO
                    if "image" in c:
                        buffered = BytesIO()
                        c["image"].save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_str}})
                    elif "url" in c:
                        parts.append({"text": f"[Image Link: {c['url']}]"})
            contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": parts})

        payload = {"contents": contents}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            result = resp.json()
            if "error" in result:
                return f"Gemini Error: {result['error']['message']}"
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Gemini Error: {e}"

    def embed(self, text: str, model_name: str = "gemini-embedding-2-preview") -> List[float]:
        """Multimodal Embedding 2.0 support for Gemini."""
        if self.provider != "gemini":
            # Fallback to local embedder via ModelRegistry
            embedder = ModelRegistry.get_embedder(DEFAULT_EMBEDDER, self.device_str)
            return embedder.encode(text).tolist()
            
        from settings_manager import SettingsManager
        settings = SettingsManager.load()
        key = settings.get("gemini_api_key")
        if not key: return []
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent?key={key}"
        payload = {
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": text}]}
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            return resp.json()["embedding"]["values"]
        except Exception:
            return []

    def _clean_output(self, text: str) -> str:
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.replace('<think>', '').replace('</think>', '')
        return text.strip()

    def get_emotions(self, text: str) -> Dict[str, float]:
        # Always use local model for emotions
        if not hasattr(self, 'model') or self.model is None:
            self.model, self.processor = ModelRegistry.get_model_and_processor(DEFAULT_SENTIMENT, model_type="classification")
            self.device = self.model.device
            
        inputs = self.processor(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        labels = ["joy", "trust", "anticipation", "surprise", "anger", "sadness", "fear", "disgust"]
        probs = torch.softmax(logits[0][:8], dim=0)
        return {label: prob.item() for label, prob in zip(labels, probs)}

DEFAULT_GENERATOR = "Qwen/Qwen3.5-0.8B"
DEFAULT_SENTIMENT = "ayjays132/EMOTIONVERSE-2"
DEFAULT_EMBEDDER = "google/embeddinggemma-300m"
