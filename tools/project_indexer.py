import os
import ast
import json
import logging
from typing import Dict, Any, List
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class ProjectIndexerTool(BaseTool):
    """
    🏢 PROJECT INDEXER: Maps the entire codebase context.
    Extracts function definitions, classes, and import dependencies using AST.
    """
    name = "project_indexer"
    description = "Scans the entire workspace to build a symbol map and dependency graph. Crucial for understanding how code changes ripple through the project."
    parameters = {
        "type": "object",
        "properties": {
            "include_tests": {"type": "boolean", "default": False, "description": "Whether to index files in the tests/ directory."}
        }
    }

    def execute(self, include_tests: bool = False, **kwargs) -> str:
        project_map = {}
        dependency_graph = {}
        
        try:
            for root, dirs, files in os.walk("."):
                # Prune noisy directories
                if any(d in root for d in ["__pycache__", ".git", ".selfresearch", "models", "exports"]):
                    continue
                if not include_tests and "tests" in root:
                    continue
                
                for f in files:
                    if f.endswith(".py"):
                        path = os.path.join(root, f)
                        file_info = self._parse_file(path)
                        project_map[path] = file_info
                        dependency_graph[path] = file_info.get("imports", [])

            res = {
                "status": "success",
                "total_files_indexed": len(project_map),
                "workspace_symbols": project_map,
                "dependency_graph": dependency_graph,
                "architecture_summary": "Substrate mapped via AST Analysis."
            }
            log.info(f"ProjectIndexer: Successfully mapped {len(project_map)} files.")
            return json.dumps(res, indent=2)
            
        except Exception as e:
            log.error(f"ProjectIndexer Error: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def _parse_file(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        info = {
            "classes": [],
            "functions": [],
            "imports": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                # Filter for top-level functions
                if isinstance(getattr(node, 'parent', None), ast.Module) or True: # ast.walk doesn't give parents easily
                    info["functions"].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                info["imports"].append(node.module)
                
        return info
