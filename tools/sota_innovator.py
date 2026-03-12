import os
from tools.base_tool import BaseTool
import json
import logging

log = logging.getLogger(__name__)

class SOTAInnovatorTool(BaseTool):
    """
    Allows the agent to formally submit, retrieve, or search SOTA innovations.
    Automatically benchmarks and compiles technical payloads before registration.
    """
    name = "sota_innovator"
    description = "Interacts with the Global SOTA Learner cache. Register breakthrough logic (benchmarked via ValidatorCompiler), retrieve by name, or search via Semantic RAG similarity."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["register", "retrieve", "search", "list"], "description": "Action to perform."},
            "concept_name": {"type": "string", "description": "Name of the innovation. Required for register/retrieve."},
            "technical_payload": {"type": "string", "description": "The code, math, or theory. Code is automatically benchmarked."},
            "domain": {"type": "string", "description": "Domain (e.g., 'Optimization')."},
            "query": {"type": "string", "description": "Search query for semantic RAG lookup."},
            "base_score": {"type": "number", "description": "Human/Agent assessment score (0-100)."},
            "force_update": {"type": "boolean", "description": "Force overwrite of existing innovation even if the new score is lower."}
        },
        "required": ["action"]
    }

    def __init__(self, os_instance=None):
        self.os = os_instance

    def execute(self, action: str, concept_name: str = "", technical_payload: str = "", domain: str = "General", query: str = "", base_score: float = 50.0, force_update: bool = False, **kwargs) -> str:
        from memory.sota_learner import SOTALearner
        from train.validator_compiler import ValidatorCompiler
        from theme_manager import theme_console, ThemeEngine
        
        learner = SOTALearner(device=self.os.device if self.os else 'cpu')
        compiler = ValidatorCompiler()
        
        try:
            if action == "register":
                if not concept_name or not technical_payload:
                    return "Error: name and payload required for registration."
                
                # Check for code blocks to benchmark
                import re
                code_blocks = re.findall(r"```python\n(.*?)\n```", technical_payload, re.DOTALL)
                
                final_score = base_score
                bench_report = ""
                
                if code_blocks:
                    # Log to system console rather than raw print
                    log.info(f"COMPILER: Benchmarking SOTA Code Logic for '{concept_name}'...")
                    bench_res = compiler.benchmark_code(code_blocks[0])
                    final_score = compiler.compute_sota_score(bench_res, base_score)
                    
                    status_color = ThemeEngine.PRIMARY if bench_res['success'] else ThemeEngine.DANGER
                    status_text = 'PASS' if bench_res['success'] else 'FAIL'
                    bench_report = f"\n[Compiler Report]: [{status_color}]{status_text}[/{status_color}] | Execution Time: {bench_res['execution_time']}"
                    
                    if not bench_res['success']:
                        bench_report += f"\n[Stack Trace]: {bench_res['error'][:200]}..."
                
                res = learner.register_innovation(concept_name, domain, technical_payload, final_score, force_update=force_update)
                return f"{res}{bench_report}"
            
            elif action == "retrieve":
                data = learner.retrieve_innovation(concept_name)
                if not data:
                    return f"Error: Innovation '{concept_name}' not found."

                # High-Fidelity Rendering
                import re
                from rich.panel import Panel
                from rich.syntax import Syntax
                from rich.markdown import Markdown
                from rich.console import Group
                from theme_manager import ThemeEngine, theme_console

                parts = re.split(r"(```python\n.*?\n```)", data["content"], flags=re.DOTALL)
                renderables = []

                for part in parts:
                    if part.startswith("```python"):
                        code = part.replace("```python\n", "").replace("```", "").strip()
                        renderables.append(Syntax(code, "python", theme="monokai", line_numbers=True))
                    elif part.strip():
                        renderables.append(Markdown(part.strip()))

                import io
                buf = io.StringIO()
                # Use theme_console configuration but redirected to buffer
                from rich.console import Console as InnerConsole
                inner = InnerConsole(file=buf, force_terminal=True, width=80, theme=ThemeEngine.get_theme())

                title = f"[os.header_accent]SOTA CONCEPT: {data['name']}[/os.header_accent] | v{data['version']} | {data['domain']}"
                inner.print(Panel(Group(*renderables), title=title, border_style=ThemeEngine.PRIMARY))

                return buf.getvalue()

            
            elif action == "search":
                if not query: return "Error: query required for search."
                matches = learner.find_similar_innovation(query)
                if matches:
                    return f"Semantic Matches for '{query}':\n" + "\n".join([f"🔸 {m}" for m in matches])
                return "No similar concepts found in the latent cache."
            
            elif action == "list":
                return learner.list_innovations(domain_filter=domain if domain != "General" else None)
                
            return "Unknown action."
        except Exception as e:
            log.error(f"SOTA Innovator Error: {e}")
            return f"SOTA Innovator Error: {str(e)}"
