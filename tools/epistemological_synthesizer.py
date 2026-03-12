from tools.base_tool import BaseTool
from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR

class EpistemologicalSynthesizerTool(BaseTool):
    name = "epistemological_synthesizer"
    description = "A cognitive tool that generates a radically new paradigm by forcing a cross-domain conceptual merger between the current research topic and an autonomously selected, orthogonal scientific or mathematical lens. Use this to break out of conventional thinking and discover novel scientific hypotheses."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The current research topic or problem you are working on."
            }
        },
        "required": ["topic"]
    }

    def __init__(self):
        self.vlm = None

    def execute(self, topic: str, **kwargs) -> str:
        if self.vlm is None:
            self.vlm = LanguageModelWrapper(model_name=DEFAULT_GENERATOR)
            
        # Dynamically select an orthogonal lens
        lens_prompt = (
            f"Given the research topic '{topic}', select a highly advanced, completely orthogonal scientific, mathematical, or systemic lens "
            f"(e.g., Information Theory, Non-equilibrium Thermodynamics, Autopoiesis, Category Theory) that is rarely applied to it. "
            f"Output ONLY the name of the lens."
        )
        chosen_lens = self.vlm.generate(lens_prompt, max_new_tokens=15, temperature=0.8).strip()
        
        prompt = (
            f"You are the Epistemological Synthesizer. Force a rigorous scientific connection between the concept of '{topic}' "
            f"and the orthogonal lens of '{chosen_lens}'.\n"
            f"Generate a single, profound, and highly technical paragraph explaining a new theoretical framework, hypothesis, or insight that emerges from this forced combination. It must be scientifically grounded, not mere science fiction.\n"
            f"Insight:"
        )
        
        insight = self.vlm.generate(prompt, max_new_tokens=200, temperature=0.7).strip()
        return f"[Synthesized Lens: {chosen_lens}]\nInsight: {insight}"
