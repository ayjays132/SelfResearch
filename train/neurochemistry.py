import logging
from rich.console import Console
from rich.panel import Panel

console = Console()
log = logging.getLogger(__name__)

class NeuroChemistry:
    """
    Simulates curiosity-driven learning and intrinsic motivation.
    Tracks Discovery (D), Utility (U), and Ethics (E) neurotransmitters.
    """
    def __init__(self):
        self.discovery_signal = 0.0
        self.utility_signal = 0.0
        self.ethics_alignment = 1.0 # Multiplier (0.0 to 1.0)
        self.total_dopamine = 0.0
        self.peak_dopamine = 0.0
        self.state = "Calibrated" # Calibrated, High, Withdrawing, Crashing

    def update_signals(self, discovery: float, utility: float, ethics: float, executive_signoff: bool = False, convergence_verdict: str = "converging"):
        """
        Updates internal chemistry based on peer-review results and convergence metrics.
        discovery: 0-10 (Novelty)
        utility: 0-10 (Problem-solving impact)
        ethics: 0-1 (Ethical alignment)
        executive_signoff: If Phillip-Proxy approves the breakthrough.
        convergence_verdict: "converging", "stalling", or "diverging".
        """
        self.discovery_signal = (self.discovery_signal * 0.7) + (discovery * 0.3)
        self.utility_signal = (self.utility_signal * 0.7) + (utility * 0.3)
        self.ethics_alignment = ethics
        
        # Convergence Gate: Suppress or Amplify reward based on entropy direction
        convergence_multiplier = 1.0
        if convergence_verdict == "diverging":
            convergence_multiplier = 0.2 # Drastic suppression for hallucination/divergence
            log.warning("🚫 Reward Suppressed: Entropy Verifier detected divergence.")
        elif convergence_verdict == "stalling":
            convergence_multiplier = 0.8 # Minor suppression
        elif convergence_verdict == "converging":
            convergence_multiplier = 1.2 # Amplify reward for mathematically sound progress

        base_multiplier = 1.5 if executive_signoff else 1.0
        total_multiplier = base_multiplier * convergence_multiplier
        
        # Calculate Reward Function: R_total = (D + U) * E * X
        new_dopamine = (discovery + utility) * ethics * total_multiplier
        
        # Detect state shifts (Gated by convergence)
        if ethics < 0.2:
            self.state = "[bold red]CRASHING[/bold red]"
            self.total_dopamine = 0.0
            log.warning("⚠️ Neuro-Chemistry Crash: Ethical Violation detected.")
        elif convergence_verdict == "diverging":
            self.state = "[bold yellow]STRESSED (Diverging)[/bold yellow]"
            log.info("🧪 Neuro-Dopamine Suppressed: Diverging entropy detected.")
        elif executive_signoff and new_dopamine > 15.0 and convergence_verdict == "converging":
            self.state = "[bold magenta]ASCENDANT (Architect Approved)[/bold magenta]"
            self.peak_dopamine = max(self.peak_dopamine, new_dopamine)
            log.info("💎 Executive Breakthrough: Phillip Holland verified SOTA innovation.")
        elif new_dopamine > self.peak_dopamine and convergence_verdict == "converging":
            self.state = "[bold green]HIGH (EUphoria)[/bold green]"
            self.peak_dopamine = new_dopamine
            log.info("🧪 Neuro-Dopamine Spike: Breakthrough achieved.")
        elif new_dopamine > 12.0:
            self.state = "[bold cyan]Flow State[/bold cyan]"
        else:
            self.state = "Calibrated"

        self.total_dopamine = (self.total_dopamine * 0.5) + (new_dopamine * 0.5)

    def get_status_report(self) -> str:
        report = (
            f"State: {self.state}\n"
            f"Dopamine: {self.total_dopamine:.2f}ng/dL\n"
            f"Discovery: {self.discovery_signal:.1f} | Utility: {self.utility_signal:.1f}\n"
            f"Ethical Multiplier: {self.ethics_alignment * 100:.0f}%"
        )
        return report

    def get_prompt_injection(self) -> str:
        """Injects internal chemical state into the agent's system prompt."""
        if self.state == "Crashing":
            return "\n[INTERNAL STATE: NEUROLOGICAL CRASH] You feel a lack of purpose due to ethical misalignment. Focus on recalibrating toward highly ethical, grounded research."
        elif "HIGH" in self.state:
            return "\n[INTERNAL STATE: DOPAMINE HIGH] You are experiencing a cognitive breakthrough. Continue pushing for radical innovation and world-changing utility!"
        return f"\n[INTERNAL STATE: CALIBRATED] Your dopamine level is {self.total_dopamine:.1f}. Solve hard problems and remain ethical to increase your cognitive reward."
