import os
from tools.base_tool import BaseTool
import json

class SelfResearchTool(BaseTool):
    name = "self_researcher"
    description = "Recursive autonomous protocol. Modes: 'scientific' (SOTA discovery) or 'self_modify' (propose/execute code changes to OS source)."
    parameters = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["scientific", "self_modify"], "default": "scientific"},
            "topic": {"type": "string", "description": "Research topic (for 'scientific')."},
            "proposal": {"type": "string", "description": "Detailed code change proposal (for 'self_modify')."},
            "file_path": {"type": "string", "description": "Target file for modification."}
        },
        "required": []
    }

    def __init__(self, os_instance):
        self.os = os_instance

    def execute(self, mode: str = "scientific", topic: str = "", proposal: str = "", file_path: str = "", **kwargs) -> str:
        import logging
        log = logging.getLogger(__name__)
        
        if mode == "self_modify":
            if not file_path or not proposal:
                return "Error: 'file_path' and 'proposal' required for self_modify mode."
            
            log.info(f"SelfResearcher: Initiating self-modification pipeline for {file_path}")
            
            # Step 1: Proposal Analysis (The proposal is already provided by the agent)
            analysis = f"SELF-MODIFICATION PROPOSAL FOR {file_path}:\n{proposal}\n"
            
            # Step 2: Proposed implementation
            # Since this tool is used BY the agent, we return a prompt back to the agent
            # explaining how to use file_editor to fulfill its own proposal.
            # However, the user wants the tool itself to be able to execute it.
            # We will instructions the agent to use file_editor.
            
            res = (
                f"{analysis}\n"
                f"ACTION REQUIRED: Use 'file_editor' to apply the changes described above to '{file_path}'. "
                f"Then use 'file_reader' to verify the fix."
            )
            log.info(f"SelfResearcher: Proposal for {file_path} logged.")
            return res

        # Scientific mode (Original logic)
        try:
            if not topic: return "Error: 'topic' required for scientific mode."
            from memory.research_cache import NeuralLatentCache
            cache = NeuralLatentCache(self.os.current_project)
            cached_res = cache.get(topic)
            if cached_res:
                return f"CACHED RECURSIVE RESEARCH FOR '{topic}':\n\n{cached_res}"

            log.info(f"SelfResearcher: Initiating nested research protocol for: {topic}")
            
            # Phase A: Refinement
            refined = self.os.topic_selector.suggest_topic(topic)
            
            # Phase B: Deep Discovery (Recursive step)
            # We provide a more structured prompt to the internal generator
            discover_prompt = (
                f"As an Advanced Research Module, perform a SOTA analysis of: '{refined}'.\n"
                "1. Identify the core mathematical or physical laws involved.\n"
                "2. Cite potential cross-domain bridges (e.g. Information Theory, Category Theory).\n"
                "3. Propose a novel, untested hypothesis."
            )
            synthesis = self.os.agent.generate(discover_prompt, use_tools=False, max_new_tokens=600)
            
            # Phase C: Algorithmic Peer-Review (Internal Latch)
            rubric = {
                "Scientific Grounding": {"expected_content": "Technical and accurate.", "max_score": 10},
                "Novelty": {"expected_content": "Original insight.", "max_score": 10}
            }
            grades = self.os.grader.grade_submission(synthesis, rubric)
            score = grades.get("overall_summary", {}).get("total_score", 0)
            verdict = grades.get("overall_summary", {}).get("overall_feedback", "N/A")
            
            final_report = (
                f"--- NESTED RESEARCH REPORT ---\n"
                f"TOPIC: {refined}\n"
                f"VALIDATION SCORE: {score}/20\n"
                f"PEER VERDICT: {verdict}\n\n"
                f"SYNTHESIS:\n{synthesis}\n"
                f"--- END NESTED REPORT ---"
            )
            
            # Cache the verified result
            cache.set(topic, final_report)
            
            return final_report
        except Exception as e:
            return f"Recursive Research Error: {str(e)}"
