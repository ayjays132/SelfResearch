from tools.base_tool import BaseTool
import json
import csv
import os
import pandas as pd
from typing import Optional, List, Any, Dict

class DatasetAnalystTool(BaseTool):
    name = "dataset_analyst"
    description = "Analyzes a scientific dataset (CSV/JSON). Can calculate stats (mean, median), filter rows, or summarize columns. Provide file_path and operation."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The path to the dataset file."},
            "operation": {
                "type": "string", 
                "enum": ["describe", "stats", "filter", "head"],
                "description": "Operation to perform."
            },
            "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to target (optional)."},
            "query": {"type": "string", "description": "Filter query (e.g., 'energy > 10') (optional)."}
        },
        "required": ["file_path", "operation"]
    }

    def execute(self, file_path: str, operation: str, columns: Optional[List[str]] = None, query: Optional[str] = None, **kwargs) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return "Error: Unsupported file format. Use CSV or JSON."
            
            if columns:
                available = [c for c in columns if c in df.columns]
                if available: df = df[available]

            if operation == "head":
                return df.head(5).to_markdown()
            
            elif operation == "describe":
                return df.describe().to_markdown()
            
            elif operation == "stats":
                stats = df.agg(['mean', 'median', 'std', 'min', 'max']).to_dict()
                return json.dumps(stats, indent=2)
            
            elif operation == "filter" and query:
                filtered = df.query(query)
                return f"Filtered Result (showing first 5):\n{filtered.head(5).to_markdown()}"
            
            return f"Unknown or incomplete operation '{operation}'."
            
        except Exception as e:
            return f"Error analyzing dataset: {str(e)}"
