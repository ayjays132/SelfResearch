import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel

from tools.base_tool import BaseTool
from tools.calculator import CalculatorTool
from tools.epistemological_synthesizer import EpistemologicalSynthesizerTool
from tools.web_search import WebSearchTool
from tools.unit_converter import ScientificUnitConverterTool
from tools.dataset_analyst import DatasetAnalystTool
from tools.scientific_plotter import ScientificPlotterTool
from tools.entropy_verifier import EntropyVerifier
from tools.iteration_tracker import IterationTracker
from tools.visual_browser import VisualBrowserTool
from tools.model_maker_tools import ModelArchitectTool, WorkspaceUtilityTool, WeightMapperTool, ModelBuilderTool, ShellExecutionTool
from tools.developer_tools import FileReaderTool, FileEditorTool, TodoManagerTool, AskUserTool, CornellNotesTool, VisionInspectorTool
from tools.workspace_a11y import WorkspaceA11yTreeTool
from tools.empirical_generator import EmpiricalGeneratorTool
from tools.gemini_tools import GeminiImageGeneratorTool
from tools.self_research_tool import SelfResearchTool
from tools.simulation_tool import SimulationTool
from tools.academic_search_tool import AcademicSearchTool
from tools.sandbox_executor import SandboxExecutorTool
from tools.export_tool import ExportLayerTool
from tools.project_indexer import ProjectIndexerTool
from tools.test_runner import TestRunnerTool
from tools.syntax_checker import SyntaxCheckerTool
from tools.weight_transfer_tool import WeightTransferTool
from tools.hybrid_initializer import HybridInitializerTool
from tools.coherence_evaluator import CoherenceEvaluatorTool
from tools.tokenizer_offset_manager import TokenizerOffsetManager

log = logging.getLogger(__name__)
console = Console()

class ToolManager:
    """Manages the registration, prompt injection, and execution of AI tools."""
    def __init__(self, os_instance=None):
        self.os = os_instance
        self.all_tools: Dict[str, BaseTool] = {}
        self.active_tool_names: List[str] = []

        # Base tools
        self._register(CalculatorTool())
        self._register(WebSearchTool())
        self._register(EpistemologicalSynthesizerTool())
        self._register(ScientificUnitConverterTool())
        self._register(DatasetAnalystTool())
        self._register(ScientificPlotterTool())
        self._register(EntropyVerifier())
        self._register(SimulationTool())
        self._register(IterationTracker(rag_instance=self.os.rag if (self.os and hasattr(self.os, 'rag')) else None))
        self._register(VisualBrowserTool())
        self._register(AcademicSearchTool())
        self._register(ExportLayerTool())
        self._register(ProjectIndexerTool())
        self._register(TestRunnerTool())
        self._register(SyntaxCheckerTool())
        self._register(WeightTransferTool())
        self._register(HybridInitializerTool())
        self._register(CoherenceEvaluatorTool())
        self._register(TokenizerOffsetManager())
        
        # Specialized tools
        self._register(ModelArchitectTool())
        self._register(WorkspaceUtilityTool())
        self._register(WeightMapperTool())
        self._register(ModelBuilderTool())
        self._register(ShellExecutionTool())
        self._register(SandboxExecutorTool())
        
        # Developer tools
        self._register(FileReaderTool())
        self._register(FileEditorTool())
        self._register(TodoManagerTool())
        self._register(AskUserTool(os_instance=self.os))
        self._register(CornellNotesTool())
        self._register(VisionInspectorTool())
        self._register(WorkspaceA11yTreeTool())
        self._register(EmpiricalGeneratorTool())
        
        # Gemini-exclusive tools (Conditional)
        if self.os and self.os.settings.get("gemini_api_key"):
            self._register(GeminiImageGeneratorTool(os_instance=self.os))

        if self.os:
            self._register(SelfResearchTool(self.os))

    def _register(self, tool: BaseTool):
        self.all_tools[tool.name] = tool

    def set_mode(self, mode: str):
        """Configure tool subset for specific modes."""
        gemini_tools = ["nano_banana"] if "nano_banana" in self.all_tools else []
        
        if mode == "Research":
            self.active_tool_names = ["calculator", "web_search", "academic_search", "epistemological_synthesizer", "unit_converter", "dataset_analyst", "scientific_plotter", "visual_browser", "workspace_a11y_tree", "empirical_generator", "simulation_lab", "export_layer", "project_indexer", "syntax_checker", "weight_transfer_tool", "coherence_evaluator", "tokenizer_offset_manager"] + gemini_tools
        elif mode == "Model Maker":
            self.active_tool_names = ["model_architect", "weight_mapper", "model_builder", "shell_executor", "sandbox_executor", "workspace_utils", "self_researcher", "calculator", "file_reader", "file_editor", "todo_manager", "ask_user", "cornell_notes", "vision_inspector", "workspace_a11y_tree", "academic_search", "project_indexer", "test_runner", "syntax_checker", "weight_transfer_tool", "hybrid_initializer", "coherence_evaluator", "tokenizer_offset_manager"] + gemini_tools
        elif mode == "Developer":
            self.active_tool_names = ["workspace_utils", "shell_executor", "sandbox_executor", "web_search", "self_researcher", "calculator", "file_reader", "file_editor", "todo_manager", "ask_user", "cornell_notes", "vision_inspector", "workspace_a11y_tree", "project_indexer", "test_runner", "syntax_checker"] + gemini_tools
        else: # Default/All
            self.active_tool_names = list(self.all_tools.keys())

    def get_tools_prompt(self) -> str:
        """Generates a system prompt string describing active tools."""
        active_tools = [self.all_tools[n].get_schema() for n in self.active_tool_names if n in self.all_tools]
        prompt = (
            "You are an AI Assistant with access to specialized tools. "
            "To use a tool, you MUST output a JSON block exactly like this:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "kwargs": {"arg1": "value"}}\n'
            "</tool_call>\n\n"
            "Available Tools for your current mode:\n"
            f"{json.dumps(active_tools, indent=2)}\n\n"
            "If you have the answer, provide it. If you need a tool, call it and WAIT."
        )
        return prompt

    def parse_and_execute(self, response_text: str) -> tuple[bool, str]:
        """
        Robustly parses tool calls, handling malformed JSON, missing kwargs, 
        and extra conversational text from small models.
        """
        # Look for the last <tool_call> block in case of multiple
        matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL))
        if not matches:
            return False, ""
            
        match = matches[-1]
        json_str = match.group(1).strip()
        
        try:
            # 1. Attempt standard parse
            try:
                call_data = json.loads(json_str)
            except json.JSONDecodeError:
                # 2. Fuzzy Parse: try to find the first { and last }
                start = json_str.find('{')
                end = json_str.rfind('}')
                if start != -1 and end != -1:
                    call_data = json.loads(json_str[start:end+1])
                else:
                    raise
            
            tool_name = call_data.get("name")
            if not tool_name:
                # 3. Handle model outputting {"tool_name": {"arg": "val"}}
                tool_name = next(iter(call_data))
                kwargs = call_data[tool_name]
            else:
                # Robustly handle flattened args vs 'kwargs' dict
                kwargs = call_data.get("kwargs")
                if kwargs is None:
                    kwargs = {k: v for k, v in call_data.items() if k != "name"}
            
            if tool_name not in self.all_tools:
                return True, f"<tool_result>\nError: Tool '{tool_name}' not found. Available: {list(self.all_tools.keys())}\n</tool_result>"
                
            from theme_manager import ThemeEngine, theme_console
            
            # Custom Tool Emojis and Styling
            tool_aesthetics = {
                "file_reader": {"icon": "📂", "color": "cyan"},
                "file_editor": {"icon": "📝", "color": "yellow"},
                "shell_executor": {"icon": "🖥️", "color": "magenta"},
                "web_search": {"icon": "🌐", "color": "blue"},
                "academic_search": {"icon": "🎓", "color": "purple"},
                "project_indexer": {"icon": "🏢", "color": "cyan"},
                "test_runner": {"icon": "🧪", "color": "green"},
                "syntax_checker": {"icon": "🔍", "color": "red"},
                "weight_transfer_tool": {"icon": "🏋️", "color": "orange3"},
                "hybrid_initializer": {"icon": "🧬", "color": "bright_magenta"},
                "coherence_evaluator": {"icon": "🧠", "color": "bright_cyan"},
                "simulation_lab": {"icon": "🔬", "color": "bright_green"},
                "export_layer": {"icon": "📤", "color": "white"},
            }
            aesthetic = tool_aesthetics.get(tool_name, {"icon": "🔧", "color": ThemeEngine.ACCENT})
            
            run_panel = Panel(
                f"Running: [os.tool.name]{tool_name}[/os.tool.name]\nArgs: [os.tool.args]{json.dumps(kwargs, indent=2)}[/os.tool.args]", 
                title=f"[bold {aesthetic['color']}]{aesthetic['icon']} {tool_name.upper()}[/]", 
                border_style=aesthetic['color']
            )
            if self.os:
                self.os.append_output(run_panel)
            else:
                theme_console.print(run_panel)
            
            # Ping heartbeat before long tool execution
            if self.os: self.os.ping_heartbeat()
            
            tool_instance = self.all_tools[tool_name]
            result = tool_instance.execute(**kwargs)
            
            # Ping heartbeat after tool execution
            if self.os: self.os.ping_heartbeat()
            
            res_panel = Panel(
                f"[os.workspace.text]{str(result)[:1500]}[/os.workspace.text]" + ("..." if len(str(result)) > 1500 else ""), 
                title=f"[bold {aesthetic['color']}]✅ Result[/]", 
                border_style=aesthetic['color']
            )
            if self.os:
                self.os.append_output(res_panel)
            else:
                theme_console.print(res_panel)

            if self.os:
                formatted_kwargs = []
                for k, v in (kwargs or {}).items():
                    val = str(v).replace("\\n", " ").strip()
                    if len(val) > 30:
                        val = val[:27] + "..."
                    formatted_kwargs.append(f"{k}={val}")
                args_summary = ", ".join(formatted_kwargs)
                descriptor = f\"{tool_name} ({args_summary})\" if args_summary else tool_name
                timestamp = datetime.now().strftime(\"%H:%M:%S\")
                self.os.last_tool_activity = descriptor
                self.os.guidance_indicator = f\"{tool_name.replace('_', ' ').title()} @ {timestamp}\"
            
            return True, f"<tool_result>\n{result}\n</tool_result>"
            
        except Exception as e:
            return True, f"<tool_result>\nError parsing tool call JSON: {e}. Ensure you output STRICT valid JSON.\n</tool_result>"
