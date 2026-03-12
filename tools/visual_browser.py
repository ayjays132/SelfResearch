from tools.base_tool import BaseTool
import asyncio
from typing import Optional

class VisualBrowserTool(BaseTool):
    name = "visual_browser"
    description = "Uses a Playwright-controlled headless browser with CSS injection to autonomously browse, search, and extract information. Provide a 'query' to search for."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The topic or information to visually search for."}
        },
        "required": ["query"]
    }

    def execute(self, query: str, **kwargs) -> str:
        # Since the tool manager loop runs synchronously and the browser is async,
        # we need to handle the event loop carefully.
        from digital_literacy.browser_controller import BrowserController
        import asyncio
        
        try:
            # Check if an event loop is already running
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            bc = BrowserController(headless=True)
            
            if loop and loop.is_running():
                # We are in an existing loop (e.g. main.py run_research_pipeline)
                # We need to run the coroutine and wait for it
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(bc.autonomous_search_loop(query, max_steps=4))
            else:
                return asyncio.run(bc.autonomous_search_loop(query, max_steps=4))
        except Exception as e:
            return f"Error running visual browser: {str(e)}"
