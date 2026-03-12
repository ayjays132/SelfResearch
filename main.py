"""
SelfResearch OS: Premium Agentic Research CLI
"""
from __future__ import annotations

import os
import sys

# --- SILENCE TELEMETRY SPEW BEFORE ANY IMPORTS ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import asyncio
import logging
import json
import contextlib
import io
import time
import threading
import signal
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from collections import deque

import transformers
transformers.logging.set_verbosity_error()

from rich.tree import Tree
from rich.panel import Panel
from rich.console import Console, Group
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich.box import ROUNDED

import colorama
import questionary
from questionary import Choice

import prompt_toolkit
from prompt_toolkit import Application
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import ANSI as PTANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign, ConditionalContainer, FloatContainer, Float
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout as PTLayout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.completion import WordCompleter, Completer, Completion

from theme_manager import ThemeEngine, theme_console
from models.model_wrapper import DEFAULT_GENERATOR, DEFAULT_SENTIMENT
from config.model_paths import ensure_model_exists, GLOBAL_ROOT
from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from assessment.rubric_grader import RubricGrader
from models.model_wrapper import LanguageModelWrapper
from train.genetic_optimizer import GeneticTestTimeOptimizer
from train.neurochemistry import NeuroChemistry
from peer_collab.bio_orchestrator import get_bio_orchestrator
from settings_manager import SettingsManager

# Initialize colorama
colorama.init()

# Create a dedicated console for string rendering
render_console = Console(force_terminal=True, width=120, theme=ThemeEngine.get_theme())

def signal_handler(sig, frame):
    """Ensure clean shutdown on Ctrl+C and avoid Fortran runtime aborts."""
    if 'os_instance' in globals():
        globals()['os_instance'].shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- PREMIUM LOGGING REDIRECTION ---
class LogBufferHandler(logging.Handler):
    def __init__(self, buffer_size=200):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            name = record.name.split('.')[-1].upper()
            
            # Premium Visual Structuring for Logs
            if record.levelno >= logging.ERROR:
                styled_msg = f"[bold red]![/bold red] [{level}] {name} | {msg}"
                raw_msg = f"[os.status.error]{styled_msg}[/os.status.error]"
            elif record.levelno >= logging.WARNING:
                styled_msg = f"[bold yellow]![/bold yellow] [{level}] {name} | {msg}"
                raw_msg = f"[os.status.warning]{styled_msg}[/os.status.warning]"
            else:
                styled_msg = f"[dim]•[/dim] [{level}] {name} | {msg}"
                raw_msg = f"[os.console.log]{styled_msg}[/os.console.log]"
                
            self.buffer.append(raw_msg)
        except Exception:
            self.handleError(record)

log_handler = LogBufferHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for h in root_logger.handlers[:]: root_logger.removeHandler(h)
root_logger.addHandler(log_handler)

# Redirect noisy loggers
for noisy in ["transformers", "huggingface_hub", "sentence_transformers", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def make_header() -> Panel:
    header_text = Text.assemble(
        (r"   _____ ______ __     ______ ______ ______ _____ ______  ___   ____   ______ __  __   ____   _____ ", "os.header"), "\n",
        (r"  / ___// ____// /    / ____// __  // ____// ___// ____//   | |  _ \ / ____// / / /  / __ \ / ___/ ", "os.header"), "\n",
        (r"  \__ \/ __/  / /    / /_   / /_/ // __/   \__ \/ __/  / /| | | |_) / /    / /_/ /  / / / / \__ \  ", "os.header_accent"), "\n",
        (r" ___/ / /___ / /___ / __/  / _, _/ /___  ___/ / /___ / ___ | |  _ < /    / __  /  / /_/ / ___/ /  ", "os.header_accent"), "\n",
        (r"/____/_____//_____//_/    /_/ |_/_____/ /____/_____//_/  |_|/_/ |_|\____/_/ |_|   \____/ /____/   ", "os.header"), "\n",
        ("\n", ""),
        (" SelfResearch OS v3.3.2 | ", "italic dim white"),
        ("Architected by Phillip Holland (ayjays132)", "os.branding"),
        ("\n", "")
    )
    return Panel(Align.center(header_text), border_style=ThemeEngine.PRIMARY, padding=(1, 2))

# --- COMMAND REGISTRY ---
class Command:
    def __init__(self, name: str, handler: Callable, description: str, category: str = "General", completer: Optional[Callable] = None, requires_boot: bool = True):
        self.name = name
        self.handler = handler
        self.description = description
        self.category = category
        self.completer = completer
        self.requires_boot = requires_boot

class CommandRegistry:
    def __init__(self, os_instance):
        self.os = os_instance
        self.commands: Dict[str, Command] = {}

    def add(self, name: str, handler: Callable, description: str, category: str = "General", completer: Optional[Callable] = None, requires_boot: bool = True):
        self.commands[name] = Command(name, handler, description, category, completer, requires_boot)

    def execute(self, text: str):
        parts = text.split(" ", 1)
        cmd_key = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None
        
        if cmd_key in self.commands:
            cmd = self.commands[cmd_key]
            if cmd.requires_boot and not self.os.boot_complete:
                self.os.output_buffer.append("[os.status.error]Neural Kernel initializing. Command locked.[/os.status.error]")
                return True
            cmd.handler(args)
            return True
        return False

class OSCompleter(Completer):
    def __init__(self, os_instance):
        self.os_instance = os_instance

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        if text.startswith('/') and ' ' in text:
            parts = text.split(' ')
            cmd_key = parts[0].lower()
            
            if cmd_key in self.os_instance.registry.commands:
                cmd = self.os_instance.registry.commands[cmd_key]
                if cmd_key == "/set" and len(parts) >= 3:
                    key = parts[1]
                    val_partial = parts[2] if len(parts) > 2 else ""
                    if key == "theme":
                        for v in ["dark", "light", "matrix", "high-contrast"]:
                            if v.startswith(val_partial):
                                yield Completion(v, start_position=-len(val_partial), display_meta="Theme Value")
                    elif key == "active_model_provider":
                        for v in ["local", "ollama", "openai", "gemini"]:
                            if v.startswith(val_partial):
                                yield Completion(v, start_position=-len(val_partial), display_meta="Provider Type")
                    return

                if cmd.completer:
                    arg_partial = parts[1] if len(parts) > 1 else ""
                    completions = cmd.completer(arg_partial)
                    for c in completions:
                        if isinstance(c, str):
                            if c.lower().startswith(arg_partial.lower()):
                                yield Completion(c, start_position=-len(arg_partial))
                        elif isinstance(c, Completion):
                            yield c
            return

        word_before = document.get_word_before_cursor(WORD=True)
        if word_before.startswith('/'):
            for cmd in self.os_instance.registry.commands.values():
                if cmd.name.startswith(word_before):
                    yield Completion(cmd.name, start_position=-len(word_before), display_meta=f"[{cmd.category}] {cmd.description}")
        elif word_before.startswith('@'):
            query = word_before[1:].lower()
            try:
                for f in os.listdir('.'):
                    if f.lower().startswith(query):
                        yield Completion(f"@{f}", start_position=-len(word_before), display_meta="Local File")
            except: pass
            if self.os_instance.rag:
                try:
                    for mem in self.os_instance.rag.memory_store:
                        doc_id = mem.get('id', '')
                        if doc_id.lower().startswith(query):
                            yield Completion(f"@{doc_id}", start_position=-len(word_before), display_meta="RAG Memory")
                except: pass

class SelfResearchOS:
    def __init__(self):
        self.settings = SettingsManager.load()
        self.device = self.settings.get("hardware_acceleration", "auto")
        if self.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # --- GLOBAL MEMORY PATHS ---
        self.pref_path = os.path.join(GLOBAL_ROOT, "preference_hierarchy.json")
        self.paradox_path = os.path.join(GLOBAL_ROOT, "paradox_manifold.json")
        self.axiom_path = os.path.join(GLOBAL_ROOT, "learned_axioms.json")
        self.rubric_path = os.path.join(GLOBAL_ROOT, "active_rubric.json")
        
        # Initialize default preference hierarchy if missing
        if not os.path.exists(self.pref_path):
            os.makedirs(GLOBAL_ROOT, exist_ok=True)
            default_prefs = {"epistemic_disruption": 1.5, "unified_theory": 1.2, "routine_optimization": 0.2}
            with open(self.pref_path, "w") as f: json.dump(default_prefs, f, indent=2)

        self.history: List[Dict[str, Any]] = []
        self.journal_dir = "research_journal"
        self.current_project = "Default Discovery"
        self.current_mode = "Research" 
        self.output_buffer = deque(maxlen=200)
        self.buffer_offset = 0
        self.show_console = False
        self.user_scaffold_queue = []
        os.makedirs(self.journal_dir, exist_ok=True)
        
        # Deferred initialization
        self.rag = None
        self.tool_manager = None
        self.neuro_chemistry = NeuroChemistry()
        self.consecutive_convergence_failures = 0
        
        # UI State
        self.boot_complete = False
        self.is_agent_running = False
        self.current_progress = 0
        self.progress_task = "Initializing..."
        self.main_window = None
        self._current_status = "Idle"
        self.last_activity_time = time.time()
        self.daemon_running = False
        self.latest_tidbit = ""
        self.headless_mode = os.environ.get("SELFRESEARCH_HEADLESS", "0") == "1"
        self.last_entropy_verdict = "N/A"
        self.last_tool_activity = "Idle"
        self.guidance_indicator = "Awaiting guidance"
        self.latest_hypothesis_template = ""
        self.auto_scroll = True
        self.required_models = [
            "Qwen/Qwen3.5-0.8B",
            "google/embeddinggemma-300m",
            "ayjays132/EMOTIONVERSE-2"
        ]

        # --- TUI INITIALIZATION (ONCE) ---
        self.registry = CommandRegistry(self)
        self._init_commands()
        
        self.kb = KeyBindings()
        self._setup_keybindings()
        
        hist_file = os.path.join(self.journal_dir, ".os_history")
        self.pt_history = FileHistory(hist_file)
        
        self.pt_style = PTStyle.from_dict(ThemeEngine.get_pt_style())

        self.input_field = TextArea(
            height=1,
            prompt=HTML('<style class="prompt.name"> PHIL-OS </style><style class="prompt.symbol">▶ </style>'),
            multiline=False,
            wrap_lines=False,
            history=self.pt_history,
            completer=OSCompleter(self),
            complete_while_typing=True,
            accept_handler=self.on_input_accept,
            focus_on_click=True
        )

        if self.headless_mode:
            class HeadlessApp:
                def __init__(self):
                    self.layout = None
                    self.key_bindings = None
                    self.style = None
                    self.is_running = False
                def run(self): pass
                def exit(self): self.is_running = False
            self.app = HeadlessApp()
        else:
            self.app = Application(
                layout=self._setup_pt_layout(),
                key_bindings=self.kb,
                style=self.pt_style,
                full_screen=True,
                mouse_support=True,
                refresh_interval=0.1,
                enable_page_navigation_bindings=True
            )
            self.app.layout.focus(self.input_field)
        self.genetic_optimizer = None 
        
        # Start boot in background
        self.boot_thread = threading.Thread(target=self._boot_sequence, daemon=True)
        self.boot_thread.start()
        
    @property
    def current_status(self):
        return self._current_status

    @current_status.setter
    def current_status(self, value):
        self._current_status = value
        self.ping_heartbeat()

    def ping_heartbeat(self):
        """Pings the daemon to signal life."""
        self.last_activity_time = time.time()

    def append_output(self, item):
        """Appends to buffer and resets offset for autoscroll."""
        self.output_buffer.append(item)
        if self.auto_scroll:
            self.buffer_offset = 0
        self.ping_heartbeat()

    def _init_commands(self):
        # 1. System Category (No boot required)
        self.registry.add("/help", self.show_help, "Display available commands", "System", requires_boot=False)
        self.registry.add("/status", self.show_status, "Show OS telemetry status", "System", requires_boot=False)
        self.registry.add("/settings", self.show_settings, "View OS substrate configuration", "System", requires_boot=False)
        self.registry.add("/set", self.handle_setting_update, "Update a setting (/set key value)", "System", 
                          completer=lambda partial: list(self.settings.keys()), requires_boot=False)
        self.registry.add("/hw", self.show_hardware_info, "Detailed hardware telemetry", "System", requires_boot=False)
        self.registry.add("/exit", self.shutdown, "Graceful OS shutdown", "System", requires_boot=False)
        self.registry.add("/quit", self.shutdown, "Graceful OS shutdown", "System", requires_boot=False)

        # 2. Discovery Category (Boot required)
        modes = ["Research", "General", "Creative", "Creative-VLM", "Simulation"]
        self.registry.add("/mode", self.handle_mode_switch, f"Switch protocol ({', '.join(modes)})", "Discovery",
                          completer=lambda partial: modes)
        self.registry.add("/project", self.handle_project_switch, "Change research project scope", "Discovery")
        self.registry.add("/research", self.handle_research_direct, "Trigger a specific research protocol", "Discovery")
        self.registry.add("/tools", self.show_tools, "List active research tools", "Discovery")
        self.registry.add("/stop", self.handle_stop_daemon, "Force stop the autonomous daemon loop", "Discovery")

        # 3. Memory & UI Category
        self.registry.add("/rag", self.show_rag_stats, "Inspect vector memory metrics", "Memory")
        self.registry.add("/history", self.handle_history, "List all indexed research IDs", "Memory")
        self.registry.add("/search", self.handle_rag_search, "Search internal knowledge base (/search <query>)", "Memory")
        self.registry.add("/console", self.toggle_console, "Toggle System Console visibility", "UI", requires_boot=False)
        self.registry.add("/clear", self.handle_clear_workspace, "Flush discovery workspace buffer", "UI", requires_boot=False)
        self.registry.add("/export", self.handle_export_workspace, "Export session findings to Markdown", "UI", requires_boot=False)
        self.registry.add("/theme", self.handle_theme_switch, "Switch OS visual theme", "UI",
                          completer=lambda partial: list(ThemeEngine.PALETTES.keys()), requires_boot=False)
        self.registry.add("/daemon", self.toggle_daemon, "Toggle Autonomous Run Loop (YOLO Mode)", "Discovery")
        self.registry.add("/autoscroll", self.toggle_autoscroll, "Toggle automatic workspace autoscroll", "UI", requires_boot=False)
        self.registry.add("/credits", self.show_credits, "Display OS architect information", "System", requires_boot=False)

    # --- ADVANCED COMMAND HANDLERS ---
    def handle_stop_daemon(self, args=None):
        if self.daemon_running or self.is_agent_running:
            self.daemon_running = False
            self.is_agent_running = False
            self.append_output("[bold red]🛑 DAEMON FORCE-STOPPED. User control restored.[/bold red]")
            self.current_status = "Idle"
        else:
            self.append_output("[os.status.info]Daemon is not currently running.[/os.status.info]")
    def show_credits(self, args=None):
        credits_text = Text.assemble(
            (r"   _____ ______ __     ______ ______ ______ _____ ______  ___   ____   ______ __  __   ____   _____ ", "os.header"), "\n",
            (r"  / ___// ____// /    / ____// __  // ____// ___// ____//   | |  _ \ / ____// / / /  / __ \ / ___/ ", "os.header"), "\n",
            (r"  \__ \/ __/  / /    / /_   / /_/ // __/   \__ \/ __/  / /| | | |_) / /    / /_/ /  / / / / \__ \  ", "os.header_accent"), "\n",
            (r" ___/ / /___ / /___ / __/  / _, _/ /___  ___/ / /___ / ___ | |  _ < /    / __  /  / /_/ / ___/ /  ", "os.header_accent"), "\n",
            (r"/____/_____//_____//_/    /_/ |_/_____/ /____/_____//_/  |_|/_/ |_|\____/_/ |_|   \____/ /____/   ", "os.header"), "\n",
            ("\n", ""),
            ("Developed by: ", "bold white"), ("Phillip Holland (ayjays132)", "os.branding"), ("\n", ""),
            ("Architecture: ", "bold white"), ("PhillVision Recurrent Refinement", "os.header_accent"), ("\n", ""),
            ("Substrate:    ", "bold white"), ("PyTorch / HuggingFace / FAISS", "os.secondary"), ("\n", ""),
            ("\n[italic dim]Dedicated to the pursuit of autonomous scientific discovery.[/italic dim]\n", "")
        )
        self.append_output(Panel(Align.center(credits_text), border_style="bold cyan", padding=(1, 2)))

    def handle_history(self, args=None):
        if not self.rag: return
        data = self.rag.memory_store
        if not data:
            self.append_output("[os.status.info]Knowledge base is currently empty.[/os.status.info]")
            return
        
        table = Table(title="Knowledge Base Index", expand=True)
        table.add_column("Index", style="dim", width=6)
        table.add_column("Research ID", style="cyan")
        for i, mem in enumerate(data):
            table.add_row(str(i+1), mem['id'])
        self.append_output(table)

    def handle_rag_search(self, query=None):
        if not query:
            self.append_output("[os.status.error]Usage: /search <query>[/os.status.error]")
            return
        if not self.rag: return
        
        results = self.rag.query(query, top_k=3)
        if not results:
            self.append_output(f"[os.status.warning]No relevant context found for '{query}'.[/os.status.warning]")
            return
            
        for res in results:
            self.append_output(Panel(
                res['text'][:1500] + ("..." if len(res['text']) > 1500 else ""),
                title=f"[bold cyan]🔍 MATCH: {res['id']}[/bold cyan]",
                border_style="cyan",
                box=ROUNDED
            ))

    def toggle_daemon(self, args=None):
        if self.daemon_running:
            self.daemon_running = False
            self.append_output("[os.status.warning]🛑 Autonomous Daemon Stopped.[/os.status.warning]")
        else:
            self.daemon_running = True
            self.append_output("[bold green]🚀 AUTONOMOUS DAEMON INITIATED. Self-Triggering Active.[/bold green]")
            threading.Thread(target=self._daemon_loop, daemon=True).start()

    def _daemon_loop(self):
        """Persistent background thread for heartbeat and self-triggering."""
        try:
            while self.daemon_running:
                time.sleep(10)
                
                if not self.boot_complete:
                    continue

                if self.is_agent_running:
                    # 1. Heartbeat Monitor (10800s / 3-hour stall threshold for intense deep research)
                    if time.time() - self.last_activity_time > 10800: 
                        self.append_output("[bold red]⚠️ DAEMON: Research loop stalled or crashed silently. Restarting...[/bold red]")
                        logging.warning("Daemon detected stalled thread. Force resetting agent state.")
                        self.is_agent_running = False
                    continue


                # 3. Completion Awareness
                if self._check_completion():
                    self.append_output("[bold cyan]✨ DAEMON: All completion criteria met. Terminating session.[/bold cyan]")
                    self.handle_export_workspace()
                    
                    # Try audio export summary
                    if "export_layer" in self.tool_manager.all_tools:
                        try:
                            self.tool_manager.all_tools["export_layer"].execute("latest", "audio")
                        except: pass
                    
                    self.daemon_running = False
                    self.shutdown()
                    break

                # 2. Self-Triggering Research Scheduler
                next_topic = self._determine_next_topic()
                self.append_output(f"[bold magenta]🤖 DAEMON: Auto-Triggering Next Protocol -> {next_topic}[/bold magenta]")
                self.is_agent_running = True # Set immediately to prevent re-triggering
                threading.Thread(target=self._run_logic, args=(next_topic,), daemon=True).start()

        except Exception as e:
            self.append_output(f"[os.status.error]DAEMON CRITICAL ERROR: {e}[/os.status.error]")
            logging.error(f"Daemon loop crash: {e}")
            self.daemon_running = False

    def _check_completion(self) -> bool:
        # Condition A: No unresolved paradoxes > 3.0
        paradoxes = {}
        if os.path.exists(self.paradox_path):
            try:
                with open(self.paradox_path, "r") as f: paradoxes = json.load(f)
            except: pass
        if any(w >= 3.0 for w in (paradoxes.values() if isinstance(paradoxes, dict) else [])):
            return False

        # Condition B: Last 3 consecutive loops scored < 70% with no improvement
        if self.consecutive_convergence_failures < 3:
            return False
            
        # Condition C: Preference hierarchy goals met (Approximated by having >= 3 learned axioms)
        axioms = []
        if os.path.exists(self.axiom_path):
            try:
                with open(self.axiom_path, "r") as f: axioms = json.load(f)
            except: pass
        if len(axioms) < 3:
            return False

        return True

    def _determine_next_topic(self) -> str:
        # Priority 1: Highest weight paradox
        paradoxes = {}
        if os.path.exists(self.paradox_path):
            try:
                with open(self.paradox_path, "r") as f: paradoxes = json.load(f)
            except: pass
        if isinstance(paradoxes, dict):
            obsessions = {p: w for p, w in paradoxes.items() if w >= 3.0}
            if obsessions:
                return max(obsessions, key=obsessions.get)
        
        # Priority 2: Lowest scoring topic in journal
        if "iteration_tracker" in self.tool_manager.all_tools:
            try:
                # We use 'discovery' as a general query to find previous research
                res = self.tool_manager.all_tools["iteration_tracker"].execute("discovery", track_code=False)
                metrics = res.get("research_metrics", {})
                delta = metrics.get("score_delta")
                if delta is not None and isinstance(delta, (int, float)) and delta <= 0:
                    return f"Resolve failures in: {res.get('topic', 'previous research')}"
            except: pass

        # Priority 3: Fresh radical topic from preferences
        prefs = {}
        if os.path.exists(self.pref_path):
            try:
                with open(self.pref_path, "r") as f: prefs = json.load(f)
            except: pass
        
        base = max(prefs, key=prefs.get) if prefs else "Advanced Theoretical Architectures"
        return f"Suggest a radical new hypothesis for: {base}"

    # --- ADVANCED COMMAND HANDLERS ---
    def handle_research_direct(self, topic=None):
        if not topic:
            self.append_output("[os.status.error]Usage: /research <topic>[/os.status.error]")
            return
        threading.Thread(target=self._run_logic, args=(topic,), daemon=True).start()

    def handle_clear_workspace(self, args=None):
        self.output_buffer.clear()
        self.append_output("[os.status.success]Discovery Workspace flushed.[/os.status.success]")

    def handle_export_workspace(self, args=None):
        filename = f"export_{int(time.time())}.md"
        content = f"# SelfResearch OS Export - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        for item in self.output_buffer:
            content += f"{str(item)}\n\n"
        with open(filename, "w", encoding="utf-8") as f: f.write(content)
        self.append_output(f"[os.status.success]Session exported to {filename}[/os.status.success]")

    def handle_theme_switch(self, theme=None):
        if not theme:
            self.append_output(f"[os.status.info]Available: {', '.join(ThemeEngine.PALETTES.keys())}[/os.status.info]")
            return
        theme = theme.strip().lower()
        if theme not in ThemeEngine.PALETTES:
            self.append_output(f"[os.status.error]Unknown theme '{theme}'.[/os.status.error]")
            return
            
        self.settings["theme"] = theme
        SettingsManager.save(self.settings)
        
        # Hot-swap the UI Styles
        self.pt_style = PTStyle.from_dict(ThemeEngine.get_pt_style())
        self.app.style = self.pt_style
        
        global render_console
        render_console = Console(force_terminal=True, width=render_console.width if render_console else 120, theme=ThemeEngine.get_theme())
        
        self.append_output(f"[os.status.success]Visual substrate live-recalibrated to {theme.upper()}.[/os.status.success]")

    def toggle_autoscroll(self, args=None):
        self.auto_scroll = not self.auto_scroll
        status = "ENABLED" if self.auto_scroll else "PAUSED"
        if self.auto_scroll:
            self.buffer_offset = 0
        self.append_output(f"[os.status.success]Autoscroll {status}. Use /autoscroll to toggle.[/os.status.success]")

    def _setup_keybindings(self):
        @self.kb.add('c-t')
        def _(event): self.show_console = not self.show_console
        @self.kb.add('c-l')
        def _(event): self.handle_clear_workspace()
        @self.kb.add('pageup')
        def _(event):
            self.buffer_offset = min(len(self.output_buffer) - 5, self.buffer_offset + 5)
            self.auto_scroll = False
        @self.kb.add('pagedown')
        def _(event):
            self.buffer_offset = max(0, self.buffer_offset - 5)
            if self.buffer_offset == 0:
                self.auto_scroll = True
        @self.kb.add('tab')
        def _(event):
            if event.app.layout.has_focus(self.input_field):
                event.app.layout.focus(self.main_window)
            else:
                event.app.layout.focus(self.input_field)

    def _setup_pt_layout(self):
        def get_tui_content():
            try: 
                term_size = self.app.output.get_size()
                # Subtract 4 rows for the input frame to prevent overlapping UI
                render_console.size = (term_size.columns, max(10, term_size.rows - 4))
            except: pass
            rich_layout = self.render_rich_layout()
            with render_console.capture() as capture:
                render_console.print(rich_layout)
            return PTANSI(capture.get())

        class ScrollAwareControl(FormattedTextControl):
            def mouse_handler(inner_self, mouse_event: MouseEvent):
                if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    self.buffer_offset = min(max(len(self.output_buffer) - 5, 0), self.buffer_offset + 3)
                    self.auto_scroll = False
                elif mouse_event.event_type == MouseEventType.SCROLL_UP:
                    self.buffer_offset = max(0, self.buffer_offset - 3)
                    if self.buffer_offset == 0:
                        self.auto_scroll = True
                    else:
                        self.auto_scroll = False
                return None

        self.main_window = Window(content=ScrollAwareControl(get_tui_content, focusable=True))
        root = FloatContainer(
            content=HSplit([
                self.main_window,
                Window(height=1, char=' '), # Padding
                Frame(self.input_field, title="[ PHIL-OS SUBSTRATE ]", style="class:os.border")
            ]),
            floats=[
                Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=16))
            ]
        )
        return PTLayout(root)

    def render_rich_layout(self) -> Layout:
        state = self.neuro_chemistry.state.lower()
        border_style = ThemeEngine.PRIMARY
        if "flow" in state: border_style = "bold cyan"
        elif "ascendant" in state: border_style = "bold magenta"
        elif "high" in state: border_style = "bold green"
        elif "crashing" in state: border_style = "bold red"
        elif "stressed" in state: border_style = "bold yellow"

        l = Layout()
        l.split(
            Layout(name="header", size=9),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=12, visible=self.show_console)
        )
        l["main"].split_row(
            Layout(name="sidebar_group", size=34),
            Layout(name="workspace", ratio=1)
        )
        l["sidebar_group"].split_column(
            Layout(name="sidebar", ratio=3),
            Layout(name="orchestrator", ratio=1, visible=bool(self.latest_tidbit))
        )
        
        l["header"].update(make_header())
        l["sidebar"].update(self._get_sidebar_panel(border_style))
        if self.latest_tidbit:
            l["orchestrator"].update(self._get_orchestrator_panel(border_style))
        
        if not self.boot_complete:
            l["workspace"].update(self._get_boot_panel())
        else:
            l["workspace"].update(self._get_workspace_panel(border_style))
            
        if self.show_console:
            l["footer"].update(self._get_footer_panel())
        return l

    def _get_sidebar_panel(self, border_style: str) -> Panel:
        meta_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        meta_table.add_row("[os.sidebar.label]PROJ[/os.sidebar.label]", f"[os.project]{self.current_project}[/os.project]")
        meta_table.add_row("[os.sidebar.label]MODE[/os.sidebar.label]", f"[os.mode]{self.current_mode}[/os.mode]")
        meta_table.add_row("[os.sidebar.label]HW[/os.sidebar.label]", f"[yellow]{self.device.upper()}[/yellow]")
        
        # Count Learned Axioms
        axioms_count = 0
        if os.path.exists(self.axiom_path):
            try:
                with open(self.axiom_path, "r") as f: axioms_count = len(json.load(f))
            except: pass
        meta_table.add_row("[os.sidebar.label]AXIOMS[/os.sidebar.label]", f"[bold cyan]{axioms_count}[/bold cyan]")
        auto_label = "[bold green]ON[/bold green]" if self.auto_scroll else "[bold red]OFF[/bold red]"
        meta_table.add_row("[os.sidebar.label]AUTOSCROLL[/os.sidebar.label]", auto_label)

        latest_hypo = "Awaiting Hypothesis..."
        for item in reversed(self.output_buffer):
            if "Hypothesis:" in str(item):
                latest_hypo = str(item).split("Hypothesis:")[1].strip()
                break
        cue_text = self.latest_hypothesis_template or latest_hypo
        cue_panel = Text(cue_text, style="italic dim", overflow="ellipsis", no_wrap=False)
        
        def make_bar(val: float, color: str) -> str:
            filled = int(val)
            return f"[{color}]" + "█" * filled + "░" * (10 - filled) + f"[/{color}] {val:.1f}"

        substrate_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        substrate_table.add_row("[os.sidebar.label]DISCOVERY[/os.sidebar.label]", make_bar(self.neuro_chemistry.discovery_signal, "cyan"))
        substrate_table.add_row("[os.sidebar.label]UTILITY[/os.sidebar.label]", make_bar(self.neuro_chemistry.utility_signal, "green"))
        
        intel_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        el_color = "green" if self.last_entropy_verdict == "converging" else ("red" if self.last_entropy_verdict == "diverging" else "yellow")
        intel_table.add_row("[os.sidebar.label]EL (Entropy)[/os.sidebar.label]", f"[{el_color}]{self.last_entropy_verdict.upper()}[/{el_color}]")
        gen_score = self.genetic_optimizer.best_score if self.genetic_optimizer else 0.0
        intel_table.add_row("[os.sidebar.label]COHERENCE[/os.sidebar.label]", f"[cyan]{gen_score:.1f}[/cyan]")
        ethics_val = self.neuro_chemistry.ethics_alignment * 100
        intel_table.add_row("[os.sidebar.label]ETHICS[/os.sidebar.label]", f"[magenta]{ethics_val:.0f}%[/magenta]")
        dope_val = self.neuro_chemistry.total_dopamine
        intel_table.add_row("[bold orange3]DOPAMINE[/bold orange3]", f"[bold]{dope_val:.1f}[/bold] ng")
        
        daemon_status = "[bold green]ACTIVE[/bold green]" if self.daemon_running else "[dim]INACTIVE[/dim]"
        guidance_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        guidance_table.add_row("[os.sidebar.label]GUIDED BY[/os.sidebar.label]", Text(self.last_tool_activity, overflow="ellipsis"))
        guidance_table.add_row("[os.sidebar.label]THINKING[/os.sidebar.label]", f"[os.status.info]{self.guidance_indicator}[/os.status.info]")

        return Panel(
            Group(
                "\n[os.header_accent]CORE METADATA[/os.header_accent]", meta_table,
                f"\n[os.header_accent]DAEMON STATUS:[/os.header_accent] {daemon_status}",
                "\n[os.header_accent]RESEARCH CUES[/os.header_accent]", cue_panel,
                "\n[os.header_accent]NEURAL SUBSTRATE[/os.header_accent]", substrate_table,
                "\n[os.header_accent]SOTA INTEL[/os.header_accent]", intel_table,
                "\n[os.header_accent]THINKING SYSTEM[/os.header_accent]", guidance_table,
                f"\n[bold]{self.neuro_chemistry.state}[/bold]"
            ), 
            title="[os.header_accent]PHIL-OS SUBSTRATE[/os.header_accent]", 
            border_style=border_style
        )

    def _get_orchestrator_panel(self, border_style: str) -> Panel:
        content = Text(self.latest_tidbit, style="os.orchestrator.tidbit")
        return Panel(content, title="[os.orchestrator.title]🧬 TIDBIT[/os.orchestrator.title]", border_style=border_style)

    def _get_boot_panel(self) -> Panel:
        p = Progress(
            SpinnerColumn(spinner_name="dots12", style="os.header_accent"), 
            TextColumn("[bold cyan]{task.description}"), 
            BarColumn(bar_width=40, style="os.progress.bar", complete_style="os.header_accent"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=render_console
        )
        p.add_task(self.progress_task, total=100, completed=self.current_progress)
        pulse = ["-", "--", "---", "----", "---", "--", "-"][int(time.time()*4)%7]
        kernel_logs = list(log_handler.buffer)[-5:]
        log_group = Group(*[Text(l, style="os.console.log") for l in kernel_logs])
        content = Group(
            Align.center(Text(f"\n{pulse} NEURAL SYNC ACTIVE {pulse}\n", style="bold italic cyan")),
            Align.center(p),
            "\n" + "─"*60 + "\n",
            Text("SUBSYSTEM INITIALIZATION V3.0:", style="os.header_accent"),
            log_group
        )
        return Panel(content, title="[bold white]PHIL-OS KERNEL BOOT[/bold white]", border_style="os.header_accent")

    def _get_spinner_frame(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return frames[int(time.time() * 10) % len(frames)]

    def _get_workspace_panel(self, border_style: str) -> Panel:
        buf = list(self.output_buffer)
        end = len(buf) - self.buffer_offset
        start = max(0, end - 25)
        view = buf[start:end]
        
        elements = []
        from rich.box import ROUNDED
        
        if self.daemon_running:
            elements.append(Panel(Align.center("[bold magenta]🧠 YOLO MODE ACTIVE[/bold magenta] - Autonomous discovery protocol engaged. Type /stop to halt."), border_style="magenta", box=ROUNDED))
            
        for item in view:
            if isinstance(item, (Panel, Table, Group, Tree)): 
                elements.append(item)
            elif "<tool_result>" in str(item): 
                elements.append(Panel(str(item), border_style=ThemeEngine.SECONDARY, box=ROUNDED, title="[os.status.info]RESEARCH SIGNAL[/os.status.info]"))
            elif "SCIENTIFIC DISCOVERY REPORT" in str(item):
                # Google Gemini style Card
                elements.append(Panel(str(item), border_style="bold cyan", box=ROUNDED, padding=(1, 2), title="[bold white]✨ DISCOVERY CARD[/bold white]"))
            elif "[os.status.error]" in str(item):
                # Graceful degradation Card
                elements.append(Panel(str(item), border_style="bold red", box=ROUNDED, title="[bold white]⚠️ SYSTEM DEGRADATION[/bold white]"))
            else: 
                elements.append(item)
        
        if self.is_agent_running:
            spinner = self._get_spinner_frame()
            elements.append(f"\n[bold yellow]{spinner} {self.current_status}...[/bold yellow]")
            
        # Neural Pulse in Title
        pulse = ["·", "•", "●", "•", "·"][int(time.time()*2)%5]
        title = f"[os.workspace.title]{pulse} DISCOVERY WORKSPACE {pulse}[/os.workspace.title]"
        auto_label = "ON" if self.auto_scroll else "OFF"
        subtitle = f"[os.workspace.subtitle]Steps: {len(buf)} | Scroll: {self.buffer_offset} | Auto: {auto_label} | /help[/os.workspace.subtitle]"
        return Panel(Group(*elements), title=title, subtitle=subtitle, border_style=border_style)

    def _get_footer_panel(self) -> Panel:
        logs = "\n".join(list(log_handler.buffer)[-10:])
        return Panel(logs, title="[os.console.title]SYSTEM CONSOLE (RECOVERY/DEBUG)[/os.console.title]", border_style=ThemeEngine.HIGHLIGHT)

    def on_input_accept(self, buffer):
        text = buffer.text.strip()
        if not text: return
        if not self.registry.execute(text):
            if self.is_agent_running:
                self.append_output(f"[bold cyan]👤 USER SCAFFOLDING INJECTED:[/bold cyan] {text}")
                self.user_scaffold_queue.append(text)
                buffer.text = ""
                return
            threading.Thread(target=self._run_logic, args=(text,), daemon=True).start()
        buffer.text = ""

    def _run_logic(self, user_input: str):
        if not self.boot_complete:
            self.append_output("[os.status.error]Neural Substrate not yet calibrated. Input deferred.[/os.status.error]")
            return

        self.is_agent_running = True
        self.current_status = "Synthesizing Thought"
        
        class StreamToLogger(object):
            """Strictly routes redirected output to the SYSTEM CONSOLE logger."""
            def __init__(self, logger, log_level=logging.INFO):
                self.logger, self.log_level = logger, log_level
            def write(self, buf):
                for line in buf.rstrip().splitlines(): 
                    self.logger.log(self.log_level, line.rstrip())
            def flush(self): pass
            def isatty(self): return False

        # Redirection for system noise (transformers, etc.)
        sl = StreamToLogger(logging.getLogger("os.system"))
        
        try:
            with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
                if self.current_mode == "Research": self.run_research_pipeline(user_input)
                else: self.run_general_agent_loop(user_input)
        except Exception as e:
            self.append_output(f"[os.status.error]Logic Error: {e}[/os.status.error]")
        finally:
            self.is_agent_running, self.current_status = False, "Idle"

    def _boot_sequence(self):
        self.current_status = "OS Initialization"
        
        class StreamToLogger(object):
            def __init__(self, logger, log_level=logging.INFO):
                self.logger, self.log_level = logger, log_level
            def write(self, buf):
                for line in buf.rstrip().splitlines(): self.logger.log(self.log_level, line.rstrip())
            def flush(self): pass
            def isatty(self): return False
        
        sl = StreamToLogger(logging.getLogger("os.boot"))

        def init_rag():
            from memory.persistent_rag import PersistentRAG
            self.rag = PersistentRAG(device=self.device)
        def init_tools():
            from tools.tool_manager import ToolManager
            self.tool_manager = ToolManager(os_instance=self)
            self.tool_manager.set_mode(self.current_mode)
        
        try:
            with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
                # 0. SUBSTRATE CALIBRATION (Auto-download)
                for repo in self.required_models:
                    model_name = repo.split('/')[-1]
                    self.progress_task = f"Calibrating {model_name}..."
                    self.current_status = f"Downloading {model_name}"
                    ensure_model_exists(repo)

                tasks = [
                    ("Neural Kernel", self._init_agent),
                    ("Vector Memory", init_rag),
                    ("Discovery Tools", init_tools),
                    ("Semantic RAG", lambda: setattr(self, 'topic_selector', TopicSelector(device=self.device))),
                    ("Visual Browser", lambda: setattr(self, 'evaluator', SourceEvaluator(device=self.device))),
                    ("Peer-Reviewers", lambda: setattr(self, 'grader', RubricGrader(device=self.device))),
                    ("Genetic Scaffolding", lambda: setattr(self, 'genetic_optimizer', GeneticTestTimeOptimizer(self.agent, self.grader, os_instance=self)))
                ]
                
                total = len(tasks)
                for i, (name, fn) in enumerate(tasks):
                    self.progress_task = f"Booting {name}..."
                    self.current_status, self.current_progress = f"Deploying {name}", (i / total) * 100
                    fn()
                    time.sleep(0.1)
                
                self.boot_complete, self.current_status = True, "Idle"
                if self.settings.get("enable_genetic_mutation", True): self.genetic_optimizer.start()
                
                # QOL: Welcome Message
                welcome_text = (
                    "# Welcome to SelfResearch OS v3.3.2\n"
                    "Your neural kernel is calibrated and ready for discovery.\n\n"
                    "**Quick Start:**\n"
                    "- Type a topic to start manual research.\n"
                    "- Use `/daemon` for autonomous discovery (YOLO Mode).\n"
                    "- Use `/theme <name>` to swap visual substrates.\n"
                    "- Use `Ctrl+T` to toggle the debug console.\n"
                    "- Use `Ctrl+L` to clear the workspace.\n"
                )
                self.append_output(Panel(Markdown(welcome_text), title="[bold green]INITIALIZATION SUCCESS[/bold green]", border_style="green", padding=(1, 2)))
                self.append_output("[os.status.success]OS Substrate fully calibrated. Awaiting Scientific Protocol.[/os.status.success]")
        except Exception as e:
            self.append_output(f"[os.status.error]BOOT CRITICAL FAILURE: {e}[/os.status.error]")
            logging.error(f"Boot crash: {e}")

    def _logic_sync(self, user_input: str):
        """Synchronous version of _run_logic for testing/bootstrap."""
        self.is_agent_running = True
        self.current_status = "Synthesizing Thought"
        try:
            if self.current_mode == "Research":
                self.run_research_pipeline(user_input)
            else:
                self.run_general_agent_loop(user_input)

        except Exception as e:
            self.append_output(f"[os.status.error]Logic Error: {e}[/os.status.error]")
        finally:
            self.is_agent_running = False
            self.current_status = "Idle"

    def _extract_hypothesis_title(self, structured_text: str, fallback_title: str) -> str:
        for line in structured_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            lower = cleaned.lower()
            if "hypothesis title" in lower:
                parts = cleaned.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        first_lines = [l.strip() for l in structured_text.splitlines() if l.strip()]
        if first_lines:
            candidate = first_lines[0]
            if len(candidate) > 5:
                return candidate
        return fallback_title

    def _init_agent(self):
        self.agent = LanguageModelWrapper(
            model_name=self.settings.get("active_model_name", DEFAULT_GENERATOR), 
            device=self.device, provider=self.settings.get("active_model_provider", "local")
        )
        self.agent.enable_peer_review = True
        return self.agent

    # --- PIPELINE LOGIC ---
    def run_research_pipeline(self, base_topic: str):
        # --- FEATURE 2: Paradox Obsession ---
        paradoxes = {}
        if os.path.exists(self.paradox_path):
            try:
                with open(self.paradox_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list): paradoxes = {p: 1.0 for p in data}
                    else: paradoxes = data
            except: pass
            
        obsessions = [p for p, w in paradoxes.items() if w >= 3.0]
        if obsessions:
            base_topic = max(obsessions, key=lambda k: paradoxes[k])
            self.append_output(f"[bold magenta]OBSESSION OVERRIDE:[/bold magenta] System is haunted by paradox: {base_topic[:80]}...")
            paradoxes[base_topic] += 0.5 # Deepen obsession temporarily
            with open(self.paradox_path, "w") as f: json.dump(paradoxes, f, indent=2)
            
        # --- FEATURE 1: Preference Hierarchy (Caring) ---
        self.current_status = "Evaluating Preference Hierarchy"
        preferences = {}
        if os.path.exists(self.pref_path):
            try:
                with open(self.pref_path, "r") as f: preferences = json.load(f)
            except: pass
            
        care_prompt = f"Given core preferences and weights: {json.dumps(preferences)}\nRate how much we care about: '{base_topic}'.\nReturn ONLY a float between 0.1 (apathy) and 2.5 (obsession)."
        try:
            raw_care = self.agent.generate(care_prompt, max_new_tokens=10).strip()
            care_factor = float(''.join(c for c in raw_care if c.isdigit() or c == '.'))
        except:
            care_factor = 1.0
            
        self.append_output(f"[os.status.info]Intrinsic Motivation (Care Factor):[/os.status.info] {care_factor:.2f}")
        
        if care_factor > 1.8:
            pref_prompt = f"We deeply cared about '{base_topic}'. Formulate a 1-3 word abstract goal to add to our preference hierarchy. Return ONLY the words."
            new_pref = self.agent.generate(pref_prompt, max_new_tokens=10).strip()
            if new_pref and len(new_pref) < 40:
                preferences[new_pref] = 1.0
                with open(self.pref_path, "w") as f: json.dump(preferences, f, indent=2)
                self.append_output(f"[bold cyan]Preference Evolved:[/bold cyan] {new_pref}")

        max_effort_tokens = int(1500 * max(0.5, min(2.5, care_factor)))

        # --- FEATURE 3: Embodied Feedback (Signal Acquisition) ---
        self.current_status = "Embodied Signal Acquisition"
        surprise_query = f"latest groundbreaking discovery or anomaly in {base_topic} 2025 2026"
        live_signal = self.tool_manager.all_tools["web_search"].execute(surprise_query)
        self.append_output(f"[os.status.info]LIVE SIGNAL INGESTED:[/os.status.info] {str(live_signal)[:100]}...")

        # --- FEATURE 2: Productive Misunderstanding (Alien Novelty) ---
        self.current_status = "Alien Hypothesis Generation"
        alien_prompt = (
            f"Base Topic: {base_topic}\nLive Signal: {live_signal}\n\n"
            "Task: Think like an eccentric visionary scientist. Intentionally 'misunderstand' or challenge a core assumption "
            "from the signal, then hallucinate a missing dimension or alien leverage point that explains the anomaly with unexpected depth. "
            "Use the scaffold below to build the response and lean into the template so the hypothesis is incredible, deeply technical, and ready for peer critique.\n\n"
            "Format the response EXACTLY as Markdown bullets; do not add extra commentary beyond the scaffold.\n"
            "- Hypothesis Title: A single vivid title summarizing the radical leap.\n"
            "- Observation / Signal Summary: What anomaly or data wrinkle motivated this leap?\n"
            "- Assumption Being Challenged: Which hidden belief is being flipped?\n"
            "- Novel Mechanism or Leverage Point: The unexpected causal twist that makes the hypothesis work.\n"
            "- Prediction (Quantitative/Binary): A crisp testable statement with dataset or simulation grounding.\n"
            "- Evaluation Plan: Where to look, metric to check, or simulation to run to falsify/refine the claim.\n"
        )
        structured_hypothesis = self.agent.generate(
            alien_prompt,
            max_new_tokens=220,
            temperature=0.95,
            top_p=0.95
        ).strip()
        self.latest_hypothesis_template = structured_hypothesis
        refined_topic = self._extract_hypothesis_title(structured_hypothesis, base_topic).strip()
        if not refined_topic:
            refined_topic = base_topic
        self.append_output(Panel(Text(structured_hypothesis, style="os.workspace.text"), title="[bold cyan]HYPOTHESIS SCAFFOLD[/bold cyan]", border_style=ThemeEngine.SECONDARY))
        self.append_output(f"[os.branding]Alien Hypothesis:[/os.branding] {refined_topic}")
        
        self.current_status = "Retrieving Knowledge Graph"
        past_context = ""
        if self.rag:
            docs = self.rag.query(refined_topic, top_k=2)
            if docs:
                past_context = "\n\nPAST KNOWLEDGE GRAPH:\n" + "\n---\n".join([d['text'][:800] for d in docs])
        
        # Haunting Context: Inject Unresolved Paradoxes
        if paradoxes:
            past_context += "\n\nUNRESOLVED PARADOXES (Stay confused by these):\n" + "\n".join(f"- {p}" for p in list(paradoxes.keys())[-3:])
                
        # --- FEATURE 4: Meta-Strategy (Productive Confusion) ---
        self.current_status = "Meta-Strategy Evaluation"
        meta_prompt = f"Hypothesis: {refined_topic}\n{past_context}\n\nBased on the history, should we: [GO DEEP], [PIVOT], [ABANDON], or [STAY CONFUSED]? Answer with just one word."
        strategy = self.agent.generate(meta_prompt, max_new_tokens=50).strip().upper()
        self.append_output(f"[os.status.warning]Meta-Strategy Layer:[/os.status.warning] {strategy}")
        
        if "STAY CONFUSED" in strategy:
            self.current_status = "Paradox Manifestation"
            paradox_prompt = f"Hypothesis {refined_topic} is contradictory. Formulate the core paradox into a single haunting question. Return ONLY the question."
            new_paradox = self.agent.generate(paradox_prompt, max_new_tokens=100).strip()
            if isinstance(paradoxes, dict): paradoxes[new_paradox] = 1.0
            else: paradoxes = {new_paradox: 1.0}
            with open(self.paradox_path, "w") as f: json.dump(paradoxes, f, indent=2)
            self.append_output(f"[bold magenta]Paradox Manifested:[/bold magenta] {new_paradox}")
            return

        if "ABANDON" in strategy:
            self.append_output("[bold red]Research vector abandoned.[/bold red]")
            return

        # --- NEW PHASE: Epistemological Deconstruction ---
        self.current_status = "Epistemological Synthesis"
        if "epistemological_synthesizer" in self.tool_manager.all_tools:
            epistemic_insight = self.tool_manager.all_tools["epistemological_synthesizer"].execute(topic=refined_topic)
            self.append_output(f"[os.status.info]Cross-Domain Paradigm Applied.[/os.status.info]")
            past_context += f"\n\nEPISTEMOLOGICAL SHIFT:\n{epistemic_insight}"
                
        # --- FEATURE 1: External Reality Check (Testable Prediction) ---
        self.current_status = "Predictive Modeling"
        prediction_prompt = f"Based on hypothesis '{refined_topic}', make one highly specific, numeric or binary prediction about a real-world dataset or simulation result. Return ONLY the prediction."
        prediction = self.agent.generate(prediction_prompt, max_new_tokens=100).strip()
        self.append_output(f"[os.status.info]Predictive Latch:[/os.status.info] {prediction}")

        axioms = []
        if os.path.exists(self.axiom_path):
            try:
                with open(self.axiom_path, "r") as f: axioms = json.load(f)
            except: pass
        
        system_prompt = None
        if axioms:
            system_prompt = "BEHAVIORAL AXIOMS LEARNED FROM PAST BREAKTHROUGHS:\n" + "\n".join(f"- {a}" for a in axioms[-5:])
            
        # Inject User Scaffolding
        if self.user_scaffold_queue:
            past_context += "\n\nUSER DIRECTIVES & SCAFFOLDING:\n" + "\n".join(f"- {u}" for u in self.user_scaffold_queue)
            self.user_scaffold_queue.clear()
            
        full_prompt = f"Investigate: {refined_topic}{past_context}\n\nVerification Task: Verify if '{prediction}' holds true using available data or tools."
        
        self.current_status = "Autonomous Discovery"
        agent_response = self.agent.generate(
            full_prompt, 
            use_tools=True, 
            tool_manager=self.tool_manager, 
            system_prompt=system_prompt,
            max_new_tokens=max_effort_tokens
        )
        
        # --- FEATURE: WEIGHT TRANSFER LAB (Automated Integration) ---
        if any(kw in refined_topic.lower() for kw in ["model architecture", "weight transfer", "projection"]):
            self.current_status = "Weight Transfer Projection"
            self.append_output("[os.status.info]Weight Transfer Lab: Initiating cross-architecture projection...[/os.status.info]")
            
            # 1. Attempt Transfer (Simulated Source for Research Loop)
            source_paths = ["models/Qwen3.5-0.8B/model.safetensors"]
            modality = "text+vision" if "vision" in refined_topic.lower() else "text+speech"
            
            # Execute directly via tool_manager
            self.tool_manager.all_tools["weight_transfer_tool"].execute(source_paths=source_paths, modality=modality, target_state_dict={})
            
            # 2. Verify Coherence
            self.current_status = "Coherence Verification"
            eval_res_str = self.tool_manager.all_tools["coherence_evaluator"].execute(checkpoint_path="simulated_hybrid.safetensors")
            eval_res = json.loads(eval_res_str)
            coherence_score = eval_res.get("coherence_score", 0)
            
            self.append_output(f"[bold cyan]Coherence Score:[/bold cyan] {coherence_score}/100")
            agent_response += f"\n\n### WEIGHT TRANSFER LAB RESULTS:\n{eval_res_str}"
            
            # 3. Axiom Extraction (Special Case)
            if coherence_score > 70:
                self.append_output(f"[os.status.success]Weight Projection Validated (>70%). Learning methodology...[/os.status.success]")
                axiom_prompt = f"Based on this successful weight transfer (Score: {coherence_score}), extract one methodological rule for projecting {modality} weights. Return ONLY the rule."
                lab_axiom = self.agent.generate(axiom_prompt, max_new_tokens=100).strip()
                if lab_axiom:
                    axioms.append(lab_axiom)
                    with open(self.axiom_path, "w") as f: json.dump(axioms, f, indent=2)
                    self.append_output(f"[bold magenta]Lab Axiom Learned:[/bold magenta] {lab_axiom[:50]}...")

        # Reality Check (Falsification via Search or Simulation)
        self.current_status = "Embodied Reality Check"
        reality_signal = ""
        if any(kw in refined_topic.lower() for kw in ["physics", "gravity", "population", "biological"]):
            self.append_output("[os.status.info]Triggering Simulation Lab for reality check...[/os.status.info]")
            sim_prompt = f"Generate a valid JSON tool call for 'simulation_lab' to test the prediction: {prediction}. Output ONLY the tool call."
            sim_call = self.agent.generate(sim_prompt, max_new_tokens=300)
            called, reality_signal = self.tool_manager.parse_and_execute(sim_call)
        
        if not reality_signal:
            falsify_query = f"evidence against {prediction} or {refined_topic}"
            reality_signal = self.tool_manager.all_tools["web_search"].execute(falsify_query)
            
        self.append_output(f"[os.status.success]Reality Signal Received.[/os.status.success]")
        agent_response += f"\n\n### EXTERNAL REALITY CHECK (Embodied Signal):\n**Prediction:** {prediction}\n**Reality Signal:** {reality_signal}"

        self.current_status = "Causal Inference Analysis"
        causal_prompt = f"Analyze the following synthesis and strictly separate causality from correlation. Identify at least one instance where 'A and B appear together' might be falsely interpreted as 'A causes B'.\nSynthesis: {agent_response[-800:]}"
        causal_analysis = self.agent.generate(causal_prompt, max_new_tokens=300)
        agent_response += f"\n\n### Causal Inference Pass:\n{causal_analysis}"
        self.ping_heartbeat()
        
        # --- NEW PHASE: Metric Abstraction ---
        self.current_status = "Metric Abstraction"
        metric_prompt = f"Abstract the findings into a quantifiable mathematical metric or heuristic equation. Synthesis: {agent_response[-800:]}\nReturn ONLY the formula and a 1-sentence explanation."
        metric_abstraction = self.agent.generate(metric_prompt, max_new_tokens=200)
        agent_response += f"\n\n### Metric Abstraction:\n{metric_abstraction}"
        self.ping_heartbeat()

        # --- FEATURE 5 (upgraded): Theory of Other Minds (Alien Peer Review) ---
        self.current_status = "Dialectical Dissent (Alien Mind)"
        dissent_sys_prompt = "You are an ALIEN INTELLIGENCE with a fundamentally different causal model of the universe. Your purpose is to find the one hidden human assumption that, if wrong, collapses this hypothesis. Deconstruct it from an outside-context perspective."
        alien_mind = self.agent
        if self.settings.get("gemini_api_key"):
            try:
                alien_mind = LanguageModelWrapper(provider="gemini", model_name="auto")
            except: pass
        adv_prompt = f"Tear down this human research: {agent_response[-1000:]}"
        adv_critique = alien_mind.generate(adv_prompt, max_new_tokens=400, system_prompt=dissent_sys_prompt, temperature=0.99)
        self.append_output(f"[os.status.warning]Alien Dissent Generated.[/os.status.warning]")
        rebuttal_prompt = f"An alien skeptic has attacked your research: '{adv_critique}'. Defend it with evidence or pivot the hypothesis.\nYour Research: {agent_response[-1000:]}"
        rebuttal = self.agent.generate(rebuttal_prompt, max_new_tokens=400)
        agent_response += f"\n\n### Theory of Other Minds (Alien Dissent):\n**Alien Deconstruction:** {adv_critique}\n**Synthesis/Rebuttal:** {rebuttal}"
        self.append_output(f"[os.status.success]Discovery & Dialectic Phase Complete.[/os.status.success]")
        self.ping_heartbeat()

        self.current_status = "Bio-Orchestration"
        orchestrator = get_bio_orchestrator(self.agent.model_name)
        self.latest_tidbit = orchestrator.provide_guidance(refined_topic, agent_response[-1000:])
        
        self.current_status = "Entropy Verification"
        ev_result = self.tool_manager.all_tools["entropy_verifier"].execute([10, 8, 7, 6.5])
        self.last_entropy_verdict = ev_result["verdict"]
        
        self.current_status = "Metric Evaluation"
        # --- FEATURE 1: Meta-Goal Questioning (Rubric Evolution) ---
        default_rubric = {"Rigor": {"expected_content": "Deep scientific analysis and causal separation", "max_score": 10}}
        if not os.path.exists(self.rubric_path):
            with open(self.rubric_path, "w") as f: json.dump(default_rubric, f)
            rubric = default_rubric
        else:
            with open(self.rubric_path, "r") as f: rubric = json.load(f)
            
        grades = self.grader.grade_submission(agent_response, rubric)
        summary = grades.get("overall_summary", {})
        score = summary.get('score', 0)
        max_score = summary.get('max_score', 10)
        
        if score < (max_score * 0.5) and self.last_entropy_verdict == "stalling":
            self.current_status = "Evolving Rubric"
            evo_prompt = f"The research '{refined_topic}' failed. Rewrite the rubric to prioritize emergent phenomena discovered here. Output JSON."
            new_rubric_str = self.agent.generate(evo_prompt, max_new_tokens=300)
            try:
                start, end = new_rubric_str.find('{'), new_rubric_str.rfind('}')
                if start != -1 and end != -1:
                    new_rubric = json.loads(new_rubric_str[start:end+1])
                    with open(self.rubric_path, "w") as f: json.dump(new_rubric, f)
            except: pass
        
        if score > (max_score * 0.9):
            self.current_status = "Grounding Validation"
            skeptical_prompt = f"Read this high-scoring submission and list 3 fundamental flaws.\n{agent_response[-1000:]}"
            critique = self.agent.generate(skeptical_prompt, max_new_tokens=200)
            score = score * 0.85
            summary['score'] = score
            self.append_output(f"[os.status.warning]Skeptical Grounding Applied.[/os.status.warning]")
        
        if score < (max_score * 0.7):
            self.consecutive_convergence_failures += 1
        else:
            self.consecutive_convergence_failures = 0

        self.current_status = "Neuromorphic Feedback"
        # TIGHTENED GATING: Utility signal is now directly derived from the GROUNDED score.
        scaled_score = (score / max_score) * 10.0 if max_score > 0 else 0.0
        
        # Discovery signal is linked to novelty (Care Factor), Utility is linked to Truth (Rubric Score)
        self.neuro_chemistry.update_signals(
            discovery=8.0 * care_factor, 
            utility=scaled_score, 
            ethics=1.0, 
            convergence_verdict=self.last_entropy_verdict
        )
        
        # --- FEATURE 1b: Learn from breakthroughs (Gated by Dopamine AND Truth) ---
        if self.neuro_chemistry.total_dopamine > 12.0 and self.last_entropy_verdict == "converging" and score > (max_score * 0.7):
            learning_prompt = f"Extract one single, highly specific technical rule or methodological axiom learned from this successful breakthrough. Return ONLY the rule.\n{agent_response[-500:]}"
            new_axiom = self.agent.generate(learning_prompt, max_new_tokens=100).strip()
            if new_axiom:
                axioms.append(new_axiom)
                with open(self.axiom_path, "w") as f: json.dump(axioms, f, indent=2)
                self.append_output(f"[bold magenta]New Axiom Learned:[/bold magenta] {new_axiom[:50]}...")
        
        tag = ""
        if self.last_entropy_verdict == "diverging" or score < (max_score * 0.4):
            tag = "[DEAD END / FAILED HYPOTHESIS] - DO NOT RETREAD\n"

        final_report = (
            f"# SCIENTIFIC DISCOVERY REPORT: {refined_topic}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{tag}"
            f"--------------------------------------------------\n"
            f"METADATA substrate:\n"
            f"- Prediction Latch: {prediction}\n"
            f"- Entropy Verdict: {self.last_entropy_verdict.upper()}\n"
            f"- Neuromorphic State: {self.neuro_chemistry.state} ({self.neuro_chemistry.total_dopamine:.2f}ng)\n"
            f"- Validation Score: {score:.2f} / {max_score}\n"
            f"--------------------------------------------------\n\n"
            f"## AUTONOMOUS SYNTHESIS\n"
            f"{agent_response}"
        )
        
        self.current_status = "Commiting to Memory"
        self.rag.add_document(f"{refined_topic}_{int(time.time())}", final_report)
        self.append_output(f"[os.console.title]REPORT SAVED TO RAG.[/os.console.title]")

    def show_help(self, args=None):
        table = Table(title="SelfResearch OS Command substrate", expand=True, box=ThemeEngine.BOX_STYLE if hasattr(ThemeEngine, 'BOX_STYLE') else None)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Category", style="dim")
        for cmd in sorted(self.registry.commands.values(), key=lambda x: x.category):
            table.add_row(cmd.name, cmd.description, cmd.category)
        self.append_output(table)

    def show_status(self, args=None):
        status_panel = Panel(
            f"Mode: [bold cyan]{self.current_mode}[/bold cyan]\n"
            f"Project: [bold green]{self.current_project}[/bold green]\n"
            f"Device: [bold yellow]{self.device.upper()}[/bold yellow]\n"
            f"RAG Documents: [bold]{len(self.rag.memory_store) if self.rag else 0}[/bold]",
            title="OS Status", border_style=ThemeEngine.HIGHLIGHT
        )
        self.append_output(status_panel)

    def show_tools(self, args=None):
        if not self.tool_manager: return
        tree = Tree("[os.header_accent]Discovery Tools[/os.header_accent]")
        for tool_name in self.tool_manager.active_tool_names:
            tree.add(f"[cyan]{tool_name}[/cyan]")
        self.append_output(tree)

    def handle_mode_switch(self, mode=None):
        modes = ["Research", "General", "Creative", "Creative-VLM", "Simulation"]
        if not mode:
            self.append_output(f"[os.status.info]Available Modes:[/os.status.info] {', '.join(modes)}")
            return
        mode = mode.strip().title()
        if mode in [m.title() for m in modes]:
            self.current_mode = mode
            if self.tool_manager: self.tool_manager.set_mode(self.current_mode)
            self.append_output(f"[os.status.success]Protocol shifted to: {self.current_mode}[/os.status.success]")
        else:
            self.append_output(f"[os.status.error]Invalid Mode. Choose from: {', '.join(modes)}[/os.status.error]")

    def handle_project_switch(self, proj=None):
        if proj: self.current_project = proj.strip()
        self.append_output(f"[os.status.success]Project Scope: {self.current_project}[/os.status.success]")

    def show_settings(self, args=None):
        table = Table(title="OS Substrate Configuration", expand=True)
        table.add_column("Setting", style="os.sidebar.label")
        table.add_column("Value", style="cyan")
        for k, v in self.settings.items():
            table.add_row(k, str(v))
        self.append_output(table)

    def handle_setting_update(self, args=None):
        if not args or " " not in args:
            self.append_output("[os.status.error]Usage: /set <key> <value>[/os.status.error]")
            return
        parts = args.split(" ", 1)
        key = parts[0]
        val = parts[1]
        if key in self.settings:
            orig_type = type(self.settings[key])
            try:
                if orig_type == bool: cast_val = val.lower() == "true"
                elif orig_type == int: cast_val = int(val)
                else: cast_val = val
                self.settings[key] = cast_val
                SettingsManager.save(self.settings)
                self.append_output(f"[os.status.success]Substrate updated: {key} = {cast_val}[/os.status.success]")
            except Exception as e:
                self.append_output(f"[os.status.error]Type conversion error: {e}[/os.status.error]")
        else:
            self.append_output(f"[os.status.error]Unknown setting: {key}[/os.status.error]")

    def show_rag_stats(self, args=None):
        if not self.rag: return
        self.append_output(f"[os.status.info]RAG Memory:[/os.status.info] {len(self.rag.memory_store)} vectors indexed.")

    def show_hardware_info(self, args=None):
        info = f"Device: {self.device}\n"
        if torch.cuda.is_available():
            info += f"GPU: {torch.cuda.get_device_name(0)}\n"
            info += f"VRAM: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB"
        self.append_output(Panel(info, title="Hardware Telemetry"))

    def toggle_console(self, args=None): self.show_console = not self.show_console

    def shutdown(self, args=None):
        """Gracefully shutdown the OS and its background threads."""
        self.daemon_running = False
        if hasattr(self, 'genetic_optimizer') and self.genetic_optimizer:
            self.genetic_optimizer.stop()
        
        if hasattr(self, 'app') and self.app.is_running:
            self.app.exit()
        else:
            sys.exit(0)

def main():
    """Main entry point for the SelfResearch OS CLI."""
    os_instance = SelfResearchOS()
    try:
        os_instance.app.run()
    except KeyboardInterrupt:
        os_instance.shutdown()

if __name__ == "__main__":
    main()
