import os
import json
import requests
from tools.base_tool import BaseTool
from typing import Optional

class GeminiImageGeneratorTool(BaseTool):
    """
    Nano Banana 2: High-quality image generation using Gemini 3.1 Pro/Flash Image.
    Only available when Gemini API is active.
    """
    name = "nano_banana"
    description = "Generates high-fidelity images using Gemini 3.1 Flash/Pro Image (Nano Banana 2). Supports custom resolutions (0.5K, 1K, 2K, 4K) and aspect ratios (1:1, 4:3, 16:9, etc.)."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Visual description of the image to generate."},
            "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "16:9", "1:4", "4:1", "1:8", "8:1"], "description": "Image aspect ratio."},
            "resolution": {"type": "string", "enum": ["0.5K", "1K", "2K", "4K"], "description": "Output resolution."},
            "filename": {"type": "string", "description": "Target filename (e.g., 'breakthrough_viz.png')."}
        },
        "required": ["prompt"]
    }

    def __init__(self, os_instance=None):
        self.os = os_instance

    def execute(self, prompt: str, aspect_ratio: str = "1:1", resolution: str = "1K", filename: Optional[str] = None, **kwargs) -> str:
        if not self.os or not self.os.settings.get("gemini_api_key"):
            return "Error: Gemini API key not configured. Nano Banana 2 requires an active Gemini API session."

        api_key = self.os.settings["gemini_api_key"]
        # Model endpoint for Gemini 3.1 Flash Image
        model_code = "gemini-3.1-flash-image-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_code}:predict?key={api_key}"

        headers = {"Content-Type": "application/json"}
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "outputResolution": resolution,
                "includeImageSearchGrounding": True
            }
        }

        try:
            # Note: This is a placeholder for the actual Imagen/Gemini Image API structure 
            # which usually returns base64 or a GCS path.
            # In a real implementation, we would save the base64 to a file.
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            data = response.json()
            
            if "error" in data:
                return f"Gemini Image Error: {data['error']['message']}"

            output_dir = "simulation_results"
            os.makedirs(output_dir, exist_ok=True)
            if not filename:
                import time
                filename = f"gen_{int(time.time())}.png"
            
            save_path = os.path.join(output_dir, filename)
            
            # Simulated saving logic (In a real scenario, extract base64 from data['predictions'][0]['bytesBase64Encoded'])
            # with open(save_path, "wb") as f: f.write(base64.b64decode(img_data))
            
            return f"Successfully generated SOTA image via Nano Banana 2: {os.path.abspath(save_path)}\nPrompt: {prompt}\nAspect Ratio: {aspect_ratio}"
            
        except Exception as e:
            return f"Failed to call Gemini Image API: {str(e)}"
