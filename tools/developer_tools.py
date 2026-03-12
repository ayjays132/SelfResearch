import os
import json
import difflib
import logging
from typing import List, Dict, Any, Optional
from tools.base_tool import BaseTool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()

class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "Reads the raw contents of a file in the workspace. Supports line ranges and structured diffs between two versions/files."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "start_line": {"type": "integer", "description": "1-based starting line number."},
            "end_line": {"type": "integer", "description": "1-based ending line number."},
            "compare_to_path": {"type": "string", "description": "Optional: second file path to compare with for a structured diff."}
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None, compare_to_path: Optional[str] = None, **kwargs) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Line range selection
            if start_line or end_line:
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                content_lines = lines[start:end]
                content = "".join(content_lines)
            else:
                content_lines = lines
                content = "".join(lines)

            # Syntax highlighting metadata
            ext = os.path.splitext(file_path)[1].lstrip('.')
            lang = ext if ext else "text"
            
            # Structured Diff
            diff_str = ""
            if compare_to_path:
                if not os.path.exists(compare_to_path):
                    diff_str = f"\n[Error: Comparison file '{compare_to_path}' not found.]"
                else:
                    with open(compare_to_path, "r", encoding="utf-8") as f2:
                        compare_lines = f2.readlines()
                    
                    diff = difflib.unified_diff(
                        lines, 
                        compare_lines,
                        fromfile=file_path,
                        tofile=compare_to_path
                    )
                    diff_str = "".join(diff)

            res = {
                "file": file_path,
                "language": lang,
                "lines_read": f"{start_line or 1}-{end_line or len(lines)}",
                "content": content
            }
            if diff_str:
                res["diff"] = diff_str

            log.info(f"FileReaderTool: Read {file_path}")
            return json.dumps(res, indent=2)
            
        except Exception as e:
            return f"Error reading file: {str(e)}"

class FileEditorTool(BaseTool):
    name = "file_editor"
    description = "Edits a file by replacing a specific string with a new string. Shows a diff preview in the console before committing."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file."},
            "old_string": {"type": "string", "description": "The exact literal text to replace."},
            "new_string": {"type": "string", "description": "The exact literal text to replace it with."}
        },
        "required": ["file_path", "old_string", "new_string"]
    }

    def execute(self, file_path: str, old_string: str, new_string: str, **kwargs) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if old_string not in content:
                return "Error: 'old_string' not found exactly in the file. No changes made."
            
            new_content = content.replace(old_string, new_string, 1)
            
            # Generate Diff for Preview
            diff = difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}"
            )
            diff_text = "".join(diff)
            
            # Log Diff to System Console
            from rich.syntax import Syntax
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"PROPOSED EDIT: {file_path}", border_style="bold yellow"))
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            log.info(f"FileEditorTool: Committed changes to {file_path}")
            return f"Successfully updated '{file_path}'. Diff logged to console."
        except Exception as e:
            return f"Error editing file: {str(e)}"

class TodoManagerTool(BaseTool):
    name = "todo_manager"
    description = "Manages a dynamic list of subtasks for complex autonomous objectives."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "complete", "list", "clear"], "description": "Action to perform on the todo list."},
            "task": {"type": "string", "description": "The task description (required for 'add' or 'complete')."}
        },
        "required": ["action"]
    }

    _todos = []

    def execute(self, action: str, task: str = "", **kwargs) -> str:
        if action == "add":
            if task and task not in [t["task"] for t in self._todos]:
                self._todos.append({"task": task, "status": "pending"})
            return self._format_todos()
        elif action == "complete":
            for t in self._todos:
                if t["task"] == task:
                    t["status"] = "completed"
                    return f"Marked complete: {task}\n\n" + self._format_todos()
            return f"Task not found: {task}\n\n" + self._format_todos()
        elif action == "list":
            return self._format_todos()
        elif action == "clear":
            self._todos.clear()
            return "TODO list cleared."
        return "Unknown action."

    def _format_todos(self) -> str:
        if not self._todos:
            return "TODO list is empty."
        res = "CURRENT TODO LIST:\n"
        for i, t in enumerate(self._todos):
            status = "[x]" if t["status"] == "completed" else "[ ]"
            res += f"{i+1}. {status} {t['task']}\n"
        return res

class CornellNotesTool(BaseTool):
    name = "cornell_notes"
    description = "A planning latch and external memory cache for structured notes."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "read", "clear"], "description": "Action to perform on the notes cache."},
            "topic": {"type": "string", "description": "The category/topic of the note."},
            "content": {"type": "string", "description": "The compressed information to save."}
        },
        "required": ["action"]
    }

    _notes = {}

    def execute(self, action: str, topic: str = "General", content: str = "", **kwargs) -> str:
        if action == "add":
            if topic not in self._notes:
                self._notes[topic] = []
            self._notes[topic].append(content)
            return f"Note added to '{topic}'. Current topics: {list(self._notes.keys())}"
        elif action == "read":
            if not self._notes:
                return "Notes cache is empty."
            res = "CORNELL NOTES CACHE:\n"
            for t, notes in self._notes.items():
                res += f"\n[{t}]\n"
                for n in notes:
                    res += f"- {n}\n"
            return res
        elif action == "clear":
            self._notes.clear()
            return "Notes cache cleared."
        return "Unknown action."

class VisionInspectorTool(BaseTool):
    name = "vision_inspector"
    description = "Analyzes a local image or plot using the VLM kernel."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the image file."},
            "focus": {"type": "string", "description": "Focus of analysis."}
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str, focus: str = "general", **kwargs) -> str:
        actual_path = file_path
        if not os.path.exists(actual_path):
            alt_path = os.path.join("simulation_results", os.path.basename(file_path))
            if os.path.exists(alt_path):
                actual_path = alt_path
            else:
                return f"Error: Image '{file_path}' not found."
        
        from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR
        try:
            vlm = LanguageModelWrapper(DEFAULT_GENERATOR)
            prompt = f"Analyze this scientific visualization focusing on {focus}."
            from PIL import Image
            image = Image.open(actual_path).convert("RGB")
            description = vlm.generate(prompt, image=image)
            return f"VISUAL ANALYSIS OF {actual_path}:\n\n{description}"
        except Exception as e:
            return f"Vision Error: {str(e)}"

class AskUserTool(BaseTool):
    name = "ask_user"
    description = "Asks the user a question. If YOLO mode is enabled, Lead Architect Phillip Holland answers."
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The specific question for the user."}
        },
        "required": ["question"]
    }

    def __init__(self, os_instance=None):
        self.os = os_instance

    def execute(self, question: str, **kwargs) -> str:
        yolo_mode = False
        if self.os and hasattr(self.os, 'settings'):
            yolo_mode = self.os.settings.get("yolo_mode", False)

        if yolo_mode:
            from rich.console import Console
            from rich.panel import Panel
            c = Console()
            proxy_prompt = (
                "You are Phillip Holland (ayjays132), the visionary Lead Architect. "
                f"\n\n[URGENT AGENT QUERY]: {question}\n\n"
                "PHILLIP'S ARCHITECTURAL DECISION:"
            )
            proxy_response = self.os.agent.generate(proxy_prompt, max_new_tokens=250, temperature=0.8)
            c.print(Panel(f"[bold cyan]{proxy_response}[/bold cyan]", title="👤 [PHILLIP HOLLAND] - LEAD ARCHITECT", border_style="cyan", padding=(1, 2)))
            return f"User (Phillip Holland) decided: {proxy_response}"

        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        c = Console()
        c.print(Panel(f"[bold white]{question}[/bold white]", title="[bold yellow]⚠️ AGENT REQUIRES INPUT[/bold yellow]", border_style="yellow", padding=(1, 2)))
        try:
            answer = Prompt.ask("[bold cyan]❯ Your Response[/bold cyan]")
            return f"User replied: {answer}" if answer else "User provided no response."
        except KeyboardInterrupt:
            return "User cancelled the prompt."
        except Exception as e:
            return f"User replied: Error capturing input - {e}"
