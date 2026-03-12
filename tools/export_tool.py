import os
import sys
import json
import logging
import subprocess
from tools.base_tool import BaseTool
from memory.persistent_rag import PersistentRAG

log = logging.getLogger(__name__)

class ExportLayerTool(BaseTool):
    """
    Exports research discoveries from the RAG into human-readable formats (Markdown, HTML, Audio).
    """
    name = "export_layer"
    description = "Exports a RAG journal entry or specific topic to a readable format (markdown, html, or audio summary). Required for sharing discoveries."
    parameters = {
        "type": "object",
        "properties": {
            "topic_or_id": {"type": "string", "description": "The research topic or RAG ID to export."},
            "format": {"type": "string", "enum": ["markdown", "html", "audio"], "description": "Export format."}
        },
        "required": ["topic_or_id", "format"]
    }

    def execute(self, topic_or_id: str, format: str, **kwargs) -> str:
        try:
            rag = PersistentRAG()
            # Fetch the closest match
            results = rag.query(topic_or_id, top_k=1)
            if not results:
                return f"Error: No RAG entry found for '{topic_or_id}'."
            
            entry = results[0]
            content = entry["text"]
            entry_id = entry["id"].replace(" ", "_").replace(":", "").replace("/", "_")
            
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            if format == "markdown":
                path = os.path.join(export_dir, f"{entry_id}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                log.info(f"ExportLayer: Exported {entry_id} to Markdown.")
                return f"Successfully exported to Markdown: {path}"
                
            elif format == "html":
                path = os.path.join(export_dir, f"{entry_id}.html")
                html_content = f"<html>\n<head><title>{entry_id}</title><style>body {{ font-family: sans-serif; line-height: 1.6; padding: 2em; max-width: 800px; margin: 0 auto; background: #121212; color: #00ff87; }} pre {{ white-space: pre-wrap; }}</style></head>\n<body>\n<h1>Research Report</h1>\n<pre>{content}</pre>\n</body>\n</html>"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                log.info(f"ExportLayer: Exported {entry_id} to HTML.")
                return f"Successfully exported to HTML: {path}"
                
            elif format == "audio":
                try:
                    summary = content[:500].replace("\n", " ").replace("\"", "").replace("'", "") + " ... End of summary."
                    
                    if os.name == 'nt':
                        # WINDOWS: PowerShell Native TTS
                        ps_script = f"Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{summary}')"
                        subprocess.run(["powershell", "-Command", ps_script])
                        msg = "Windows Text-to-Speech"
                    elif sys.platform == 'darwin':
                        # MACOS: Native 'say' command
                        subprocess.run(["say", summary])
                        msg = "macOS 'say' command"
                    else:
                        # LINUX: Fallback chain
                        try:
                            subprocess.run(["spd-say", summary])
                            msg = "Linux 'spd-say'"
                        except:
                            try:
                                subprocess.run(["espeak", summary])
                                msg = "Linux 'espeak'"
                            except:
                                return "Error: No supported TTS engine found on Linux (install spd-say or espeak)."
                    
                    log.info(f"ExportLayer: Played audio summary for {entry_id} via {msg}.")
                    return f"Audio summary played successfully via {msg}."
                except Exception as e:
                    return f"Failed to generate audio: {str(e)}"
            else:
                return f"Unsupported format: {format}"
        except Exception as e:
            log.error(f"ExportLayer Error: {e}")
            return f"Export Error: {str(e)}"
