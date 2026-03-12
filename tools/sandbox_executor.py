import os
import json
import shutil
import logging
import subprocess
from typing import List, Optional
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class SandboxExecutorTool(BaseTool):
    """
    Executes code in a sandbox with environment snapshotting.
    Allows rolling back self-modifications if an error occurs.
    """
    name = "sandbox_executor"
    description = "Executes a python script or shell command with versioning. Snapshots specified files before execution and supports rolling back if the execution fails or modifies files destructively."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["run", "rollback"], "description": "Action: 'run' to execute and snapshot, 'rollback' to restore."},
            "command": {"type": "string", "description": "The command to run (required for 'run')."},
            "snapshot_files": {"type": "array", "items": {"type": "string"}, "description": "List of files to snapshot (e.g., ['main.py', 'settings.json'])."}
        },
        "required": ["action"]
    }

    def execute(self, action: str, command: str = "", snapshot_files: Optional[List[str]] = None, **kwargs) -> str:
        # Use the global substrate directory for snapshots
        from models.model_wrapper import GLOBAL_ROOT
        snapshot_dir = os.path.join(GLOBAL_ROOT, "sandbox_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        manifest_path = os.path.join(snapshot_dir, "manifest.json")

        if action == "rollback":
            if not os.path.exists(manifest_path):
                return "Error: No snapshot manifest found. Cannot rollback."
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                restored = []
                for orig_path, backup_path in manifest.items():
                    if os.path.exists(backup_path):
                        os.makedirs(os.path.dirname(orig_path) or ".", exist_ok=True)
                        shutil.copy2(backup_path, orig_path)
                        restored.append(orig_path)
                log.info(f"Sandbox: Rolled back {len(restored)} files.")
                return f"Successfully rolled back {len(restored)} files: {', '.join(restored)}"
            except Exception as e:
                log.error(f"Sandbox Rollback Error: {e}")
                return f"Rollback failed: {str(e)}"

        elif action == "run":
            if not command:
                return "Error: 'command' is required for action 'run'."
            
            # 1. Snapshot
            manifest = {}
            if snapshot_files:
                for fpath in snapshot_files:
                    if os.path.exists(fpath):
                        safe_name = fpath.replace("/", "_").replace("\\", "_")
                        backup_path = os.path.join(snapshot_dir, safe_name)
                        shutil.copy2(fpath, backup_path)
                        manifest[fpath] = backup_path
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
                log.info(f"Sandbox: Snapshotted {len(manifest)} files before execution.")

            # 2. Execute
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
                res = {
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "snapshotted": list(manifest.keys())
                }
                log.info(f"Sandbox: Executed '{command}' with exit code {result.returncode}")
                return json.dumps(res, indent=2)
            except Exception as e:
                log.error(f"Sandbox Execution Error: {e}")
                return json.dumps({"error": str(e), "command": command}, indent=2)
        
        return f"Unknown action: {action}"
