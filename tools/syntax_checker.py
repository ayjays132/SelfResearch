import ast
import json
import logging
from typing import Dict, Any
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class SyntaxCheckerTool(BaseTool):
    """
    🔍 SYNTAX CHECKER: Pre-execution static analysis.
    Uses AST to catch syntax errors and basic structural flaws before committing code.
    """
    name = "syntax_checker"
    description = "Performs static analysis on a Python file to catch syntax errors and structural flaws without executing the code. Crucial for pre-commit verification."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to check."},
            "content": {"type": "string", "description": "Optional: raw content to check instead of reading from file."}
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str, content: str = "", **kwargs) -> str:
        log.info(f"SyntaxChecker: Analyzing {file_path}...")
        
        try:
            if not content:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            ast.parse(content)
            
            res = {
                "status": "valid",
                "file": file_path,
                "message": "No syntax errors detected."
            }
            log.info(f"SyntaxChecker: {file_path} is valid.")
            return json.dumps(res, indent=2)
            
        except SyntaxError as e:
            res = {
                "status": "invalid",
                "file": file_path,
                "error": str(e),
                "line": e.lineno,
                "column": e.offset,
                "text": e.text.strip() if e.text else ""
            }
            log.warning(f"SyntaxChecker: Found error in {file_path} at line {e.lineno}.")
            return json.dumps(res, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
