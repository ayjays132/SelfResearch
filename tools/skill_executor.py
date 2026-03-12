from tools.base_tool import BaseTool
import json

class SkillExecutorTool(BaseTool):
    """
    Allows the agent to view and manually request specific skills from the SkillManager
    during a complex task.
    """
    name = "skill_executor"
    description = "Interacts with the Global/Workspace Neural Skill Substrate. Use this to list available skills, retrieve a skill's exact guidelines, or forge a new skill for an unknown task."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "retrieve", "forge"], "description": "Action to perform on the Skill Substrate."},
            "task_description": {"type": "string", "description": "Description of the task to retrieve or forge a skill for."},
            "task_name": {"type": "string", "description": "A short name for the new skill (required for 'forge')."}
        },
        "required": ["action"]
    }

    def __init__(self, os_instance=None):
        self.os = os_instance

    def execute(self, action: str, task_description: str = "", task_name: str = "", **kwargs) -> str:
        if not self.os or not hasattr(self.os, 'topic_selector'):
            return "Error: OS substrate not fully loaded."
            
        skill_manager = self.os.topic_selector.skill_manager
        
        try:
            if action == "list":
                return skill_manager.display_skills()
            
            elif action == "retrieve":
                if not task_description: return "Error: task_description required."
                res = skill_manager.retrieve_relevant_skills(task_description)
                return res if res else f"No relevant skills found for: {task_description}"
                
            elif action == "forge":
                if not task_name or not task_description:
                    return "Error: task_name and task_description required to forge a new skill."
                skill_manager.forge_skill(task_name, task_description)
                return f"Successfully forged and ingested new skill: {task_name}"
                
            return "Unknown action."
        except Exception as e:
            return f"SkillExecutor Error: {str(e)}"
