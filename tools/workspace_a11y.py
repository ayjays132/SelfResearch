import os
from tools.base_tool import BaseTool

class WorkspaceA11yTreeTool(BaseTool):
    name = "workspace_a11y_tree"
    description = "Provides a deep, semantic accessibility tree of the entire project workspace. Returns file structures, sizes, and module purposes to help navigate large repositories without guessing paths."
    parameters = {
        "type": "object",
        "properties": {
            "depth": {"type": "integer", "description": "How many directory levels deep to scan (default 2)."}
        },
        "required": []
    }

    def execute(self, depth: int = 2, **kwargs) -> str:
        tree_str = "WORKSPACE SEMANTIC ACCESSIBILITY TREE:\n"
        start_path = "."
        
        for root, dirs, files in os.walk(start_path):
            # Skip hidden/cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'SelfResearch.egg-info']
            
            level = root.replace(start_path, '').count(os.sep)
            if level > depth:
                continue
                
            indent = ' ' * 4 * level
            tree_str += f"{indent}[Directory] {os.path.basename(root)}/\n"
            
            subindent = ' ' * 4 * (level + 1)
            for f in files[:15]: # Limit files per dir to avoid massive tokens
                if f.endswith('.pyc') or f.startswith('.'): continue
                
                fpath = os.path.join(root, f)
                size_kb = os.path.getsize(fpath) / 1024
                
                # Add semantic hints for specific files
                hint = ""
                if f.endswith('.py'): hint = " [Python Module]"
                elif f.endswith('.json'): hint = " [Configuration/Data]"
                elif f.endswith('.md'): hint = " [Documentation]"
                elif f.endswith('.csv'): hint = " [Dataset]"
                
                tree_str += f"{subindent}[File] {f} ({size_kb:.1f} KB){hint}\n"
                
            if len(files) > 15:
                tree_str += f"{subindent}... and {len(files)-15} more files.\n"
                
        return tree_str
