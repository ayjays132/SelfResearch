from tools.base_tool import BaseTool
import json
import random
import pandas as pd
import os

class EmpiricalGeneratorTool(BaseTool):
    name = "empirical_generator"
    description = "Generates a synthetic scientific dataset (CSV) based on a hypothesis to allow for empirical testing and visualization. Provide 'columns' and 'hypothesis'."
    parameters = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "The name of the CSV file (e.g., 'entropy_data.csv')."},
            "columns": {"type": "array", "items": {"type": "string"}, "description": "List of data columns (e.g., ['time', 'energy', 'noise'])."},
            "rows": {"type": "integer", "description": "Number of data points (default 50)."},
            "trend": {"type": "string", "enum": ["linear", "exponential", "logarithmic", "random"], "description": "The mathematical trend of the data."}
        },
        "required": ["filename", "columns"]
    }

    def execute(self, filename: str, columns: list, rows: int = 50, trend: str = "linear", **kwargs) -> str:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", filename)
        
        data = {}
        for i, col in enumerate(columns):
            if trend == "linear":
                data[col] = [j * (i+1) + random.uniform(-1, 1) for j in range(rows)]
            elif trend == "exponential":
                data[col] = [1.2 ** j + random.uniform(-0.5, 0.5) for j in range(rows)]
            elif trend == "logarithmic":
                import math
                data[col] = [math.log(j+1) + random.uniform(-0.1, 0.1) for j in range(rows)]
            else:
                data[col] = [random.uniform(0, 100) for _ in range(rows)]
        
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        
        return f"Synthetic Empirical Dataset generated successfully at {path}. Use `dataset_analyst` to verify trends."
