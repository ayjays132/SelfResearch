import os
import glob
import logging
from typing import List, Dict
from memory.persistent_rag import PersistentRAG
from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR
from theme_manager import ThemeEngine, theme_console
from rich.panel import Panel
from rich.table import Table

log = logging.getLogger(__name__)

class SkillManager:
    """
    Manages '.md' skill files across Global and Workspace scopes.
    Compresses them using PersistentRAG into memory and provides semantic retrieval.
    Features 'Skill Forge' to autonomously generate new skills.
    """
    def __init__(self, device=None):
        self.rag = PersistentRAG(device=device)
        self.generator = None # Lazy load to save memory
        
        # Determine paths
        # Global skills (where the CLI package is installed)
        self.global_skills_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        # Workspace skills (where the user is currently running the CLI)
        self.workspace_skills_dir = os.path.join(os.getcwd(), ".selfresearch", "skills")
        
        os.makedirs(self.global_skills_dir, exist_ok=True)
        os.makedirs(self.workspace_skills_dir, exist_ok=True)
        
        self.loaded_skills: Dict[str, str] = {}
        self.ingest_all_skills()

    def ingest_all_skills(self):
        """Scans global and workspace directories and encodes MD files via VAE RAG."""
        self.loaded_skills.clear()
        
        # Load Global
        global_files = glob.glob(os.path.join(self.global_skills_dir, "*.md"))
        for file_path in global_files:
            self._ingest_file(file_path, scope="GLOBAL")
            
        # Load Workspace
        workspace_files = glob.glob(os.path.join(self.workspace_skills_dir, "*.md"))
        for file_path in workspace_files:
            self._ingest_file(file_path, scope="WORKSPACE")

    def _ingest_file(self, file_path: str, scope: str):
        skill_id = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id = f"skill_{scope}_{skill_id}"
            self.rag.add_document(doc_id=doc_id, text=content)
            self.loaded_skills[doc_id] = scope
        except Exception as e:
            log.error(f"Failed to ingest {scope} skill {file_path}: {e}")

    def retrieve_relevant_skills(self, task_description: str, top_k: int = 1) -> str:
        """
        Dynamically fetches and decompresses the most relevant skill for the task.
        """
        results = self.rag.query(task_description, top_k=top_k)
        if not results:
            return ""
            
        skill_text = "\n\n".join([f"<activated_skill id='{res['id']}'>\n{res['text']}\n</activated_skill>" for res in results])
        return f"\n\n[System: The following specialized skills have been retrieved from compressed memory:]\n{skill_text}\n"

    def display_skills(self) -> str:
        """Returns a rich formatted table of all loaded skills."""
        table = Table(title="Neural Skill Substrate", border_style=ThemeEngine.PRIMARY)
        table.add_column("Skill ID", style="bold cyan")
        table.add_column("Scope", style="os.header_accent")
        
        if not self.loaded_skills:
            table.add_row("No skills loaded", "-")
        else:
            for s_id, scope in self.loaded_skills.items():
                clean_id = s_id.replace("skill_GLOBAL_", "").replace("skill_WORKSPACE_", "").replace(".md", "")
                table.add_row(clean_id, scope)
                
        import io
        from rich.console import Console as InnerConsole
        buf = io.StringIO()
        inner_console = InnerConsole(file=buf, force_terminal=True, width=80, theme=ThemeEngine.get_theme())
        inner_console.print(table)
        return buf.getvalue()

    def forge_skill(self, task_name: str, task_description: str):
        """
        Skill Forge: Uses Qwen 3.5 to dynamically generate a new markdown skill file 
        for an unknown task, saves it to the Workspace scope, and ingests it.
        """
        if self.generator is None:
            self.generator = LanguageModelWrapper(model_name=DEFAULT_GENERATOR)

        log.info(f"Forging new skill for task: {task_name}...")
        prompt = (
            f"You are an expert AI creating a specialized system prompt 'Skill'.\n"
            f"Create a markdown guide for an AI Assistant tasked with: {task_name}\n"
            f"Description: {task_description}\n\n"
            f"Format strictly as:\n"
            f"# [Skill Name]\n"
            f"## Objective\n"
            f"[...]\n"
            f"## Guidelines\n"
            f"1. [...]\n\n"
            f"Markdown Output:"
        )
        
        new_skill_md = self.generator.generate(prompt, max_new_tokens=250, temperature=0.7)
        
        filename = task_name.lower().replace(" ", "_") + "_forged.md"
        filepath = os.path.join(self.workspace_skills_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_skill_md.strip())
            
        log.info(f"Skill forged and saved to {filepath}.")
        self._ingest_file(filepath, scope="WORKSPACE")
