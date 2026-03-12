import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from typing import Optional, List, Any, Dict
from tools.base_tool import BaseTool

class ScientificPlotterTool(BaseTool):
    name = "scientific_plotter"
    description = "Generates high-fidelity scientific plots (Line, Scatter, Bar, Histogram) from data. Saves results to 'simulation_results/'."
    parameters = {
        "type": "object",
        "properties": {
            "data": {"type": "object", "description": "The data to plot (dictionary of lists or a JSON string)."},
            "plot_type": {
                "type": "string",
                "enum": ["line", "scatter", "bar", "hist"],
                "description": "Type of plot."
            },
            "title": {"type": "string", "description": "Plot title."},
            "x_label": {"type": "string", "description": "X-axis label."},
            "y_label": {"type": "string", "description": "Y-axis label."},
            "filename": {"type": "string", "description": "Output filename (e.g., 'energy_levels.png')."}
        },
        "required": ["data", "plot_type", "title"]
    }

    def execute(self, data: Any, plot_type: str, title: str, x_label: str = "X", y_label: str = "Y", filename: Optional[str] = None, **kwargs) -> str:
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            filename = f"plot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Parse data
            if isinstance(data, str):
                data = json.loads(data)
            
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 6))
            plt.style.use('bmh') # Scientific style
            
            if plot_type == "line":
                df.plot(kind='line', ax=plt.gca(), marker='o')
            elif plot_type == "scatter":
                if len(df.columns) < 2: return "Error: Scatter plot requires at least 2 columns."
                df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=plt.gca())
            elif plot_type == "bar":
                df.plot(kind='bar', ax=plt.gca())
            elif plot_type == "hist":
                df.plot(kind='hist', alpha=0.7, ax=plt.gca())
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            return f"Plot generated successfully: {os.path.abspath(filepath)}"
            
        except Exception as e:
            plt.close()
            return f"Error generating plot: {str(e)}"
