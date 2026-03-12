import logging
from rich.console import Console
from rich.panel import Panel

console = Console()
log = logging.getLogger(__name__)

_reviewer_instance = None

def get_active_reviewer(model_name: str):
    global _reviewer_instance
    if _reviewer_instance is None:
        _reviewer_instance = ActivePeerReviewer(model_name)
    return _reviewer_instance

class ActivePeerReviewer:
    """
    A zero-memory-overhead parallel peer reviewer.
    Shares the exact same base model via ModelRegistry.
    Monitors the main agent's discovery loop and injects fact-checks/insights.
    """
    def __init__(self, model_name: str):
        from models.model_wrapper import LanguageModelWrapper
        # This will hit the ModelRegistry cache, using 0 extra memory
        self.vlm = LanguageModelWrapper(model_name)
        self.vlm.enable_peer_review = False # Prevent infinite recursion
        
    def review_action(self, action_text: str, result_text: str) -> str:
        console.print("[dim cyan]🔍 Parallel Peer Reviewer fact-checking...[/dim cyan]")
        prompt = (
            "You are a Parallel Peer-Reviewer agent sharing the same neural substrate as the primary agent. "
            f"Primary Agent Action:\n{action_text}\n"
            f"Tool Result:\n{result_text}\n"
            "Analyze the result. If it is solid, provide a 1-sentence deeper cross-domain insight to guide the primary agent. "
            "If no intervention is needed, strictly output the word 'PROCEED'. Do NOT use any tools."
        )
        
        # Turn off tool use for the reviewer to prevent loop instability
        insight = self.vlm.generate(prompt, use_tools=False, max_new_tokens=100)
        
        if "PROCEED" in insight.upper() or len(insight.strip()) < 10:
            return ""
            
        # Clean the insight of any residual XML logic
        if "<tool_" in insight:
            return ""
            
        console.print(Panel(f"[italic cyan]{insight}[/italic cyan]", title="💡 Peer Reviewer Insight", border_style="cyan"))
        return insight
