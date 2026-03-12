import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class AcademicSearchTool(BaseTool):
    """
    Direct API access to ArXiv for academic research.
    Retrieves real abstracts and citations, bypassing web summarizers.
    """
    name = "academic_search"
    description = "Searches the ArXiv academic database for research papers. Returns titles, authors, URLs, and full abstracts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query (e.g., 'quantum gravity', 'federated learning')."},
            "max_results": {"type": "integer", "default": 3, "description": "Number of results to return."}
        },
        "required": ["query"]
    }

    def execute(self, query: str, max_results: int = 3, **kwargs) -> str:
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}"
            response = urllib.request.urlopen(url, timeout=15)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            results = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                link = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()
                authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "url": link,
                    "abstract": summary
                })
                
            if not results:
                return f"No academic papers found on ArXiv for the query: '{query}'."
                
            formatted = []
            for i, r in enumerate(results):
                authors_str = ", ".join(r['authors'])
                formatted.append(f"[{i+1}] Title: {r['title']}\nAuthors: {authors_str}\nURL: {r['url']}\nAbstract: {r['abstract']}")
                
            log.info(f"AcademicSearchTool: Found {len(results)} papers for '{query}'.")
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            log.error(f"Academic Search Error: {e}")
            return f"Error performing academic search: {str(e)}"
