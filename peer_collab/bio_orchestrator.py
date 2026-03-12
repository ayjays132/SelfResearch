import os
import logging
from rich.console import Console
from rich.panel import Panel
from theme_manager import ThemeEngine, theme_console

console = Console()
log = logging.getLogger(__name__)

_orchestrator_instance = None

def get_bio_orchestrator(model_name: str):
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = BioOrchestrator(model_name)
    return _orchestrator_instance

class BioOrchestrator:
    """
    Bio-Communicative Orchestrator Agent.
    Provides parallel strategic guidance and 'tidbits' to the primary agent.
    Shares the same neural substrate (0 extra memory).
    """
    def __init__(self, model_name: str):
        from models.model_wrapper import LanguageModelWrapper
        # Shared substrate via ModelRegistry
        self.vlm = LanguageModelWrapper(model_name)
        self.vlm.enable_peer_review = False 
        
    def provide_guidance(self, task: str, current_history: str) -> str:
        """Generates a high-level strategic tidbit to steer the research."""
        log.info("Bio-Orchestrator generating strategic tidbit...")
        
        prompt = (
            "You are the Bio-Communicative Orchestrator of SelfResearch OS. "
            "Your role is to act as a parallel strategic advisor to the primary Research Agent. "
            f"Current Objective: {task}\n"
            f"Recent Context: {current_history}\n\n"
            "Provide a single, extraordinary strategic tidbit (1-2 sentences) that helps the primary agent "
            "better orchestrate the discovery protocol. Focus on cross-domain synthesis, hardware efficiency, or methodology. "
            "Be brilliantly brief. Strictly output the tidbit."
        )
        
        tidbit = self.vlm.generate(prompt, use_tools=False, max_new_tokens=100)
        
        if len(tidbit.strip()) < 10 or "PROCEED" in tidbit.upper():
            return ""
            
        return tidbit.strip()

    def render_tidbit(self, tidbit: str):
        """Displays the tidbit in a themed panel."""
        if not tidbit: return
        from rich.text import Text
        content = Text(tidbit, style="os.orchestrator.tidbit")
        theme_console.print(Panel(content, title="[os.orchestrator.title]🧬 Bio-Orchestrator Tidbit[/os.orchestrator.title]", border_style=ThemeEngine.ORCHESTRATOR))
