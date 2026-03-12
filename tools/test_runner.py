import os
import subprocess
import json
import logging
from typing import List, Optional
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class TestRunnerTool(BaseTool):
    """
    🧪 TEST RUNNER: Verifies code integrity via automated testing.
    Integrates with pytest to capture passes, failures, and errors.
    """
    name = "test_runner"
    description = "Runs the project's test suite using pytest. Returns a structured summary of results."
    parameters = {
        "type": "object",
        "properties": {
            "test_path": {"type": "string", "description": "Specific test file or directory to run (default: 'tests/')."},
            "capture_output": {"type": "boolean", "default": True, "description": "Whether to capture stdout/stderr from tests."}
        }
    }

    def execute(self, test_path: str = "tests/", capture_output: bool = True, **kwargs) -> str:
        log.info(f"TestRunner: Initiating verification for {test_path}...")
        
        # Check if pytest is available
        try:
            cmd = ["pytest", test_path, "--json-report", "--json-report-file=none", "-q"]
            # We use a simple subprocess call first to check basic status
            result = subprocess.run(["pytest", test_path, "-v", "--no-header"], capture_output=True, text=True, timeout=180)
            
            res = {
                "status": "completed",
                "exit_code": result.returncode,
                "summary": "Tests passed" if result.returncode == 0 else "Tests failed or errors encountered",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            log.info(f"TestRunner: Verification complete. (Exit Code: {result.returncode})")
            return json.dumps(res, indent=2)
            
        except FileNotFoundError:
            return "Error: 'pytest' not found. Please install it to use the test runner."
        except Exception as e:
            log.error(f"TestRunner Error: {e}")
            return json.dumps({"status": "error", "message": str(e)})
