import json
from typing import Dict, Any, Callable

class BaseTool:
    """Base class for all tools the AI can use."""
    name: str = "BaseTool"
    description: str = "Description of the tool"
    parameters: Dict[str, Any] = {} # JSON schema for parameters

    def execute(self, **kwargs) -> str:
        raise NotImplementedError("Tool must implement execute method.")

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
