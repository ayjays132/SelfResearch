from tools.base_tool import BaseTool
from digital_literacy.source_evaluator import SourceEvaluator

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches DuckDuckGo for real-time information, research papers, or facts. Returns a list of snippets and URLs."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        "required": ["query"]
    }

    def __init__(self):
        self.evaluator = SourceEvaluator()

    def execute(self, query: str, **kwargs) -> str:
        results = self.evaluator.search_academic_api(query, api_type="duckduckgo")
        if not results:
            return "No results found."
        
        formatted = []
        for i, r in enumerate(results[:3]):
            formatted.append(f"[{i+1}] Title: {r['title']}\nSnippet: {r['authors'][0]}\nURL: {r['url']}")
            
        return "\n\n".join(formatted)
