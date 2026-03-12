import logging
import json
import os
import difflib
from typing import Dict, Any, List, Optional
from tools.base_tool import BaseTool
from memory.persistent_rag import PersistentRAG

log = logging.getLogger(__name__)

class IterationTracker(BaseTool):
    """
    Compares consecutive research entries and tracks code change deltas between sessions.
    """
    name = "iteration_tracker"
    description = "Compares consecutive journal entries and tracks code deltas (changes in workspace files)."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The research topic to track."},
            "track_code": {"type": "boolean", "default": True, "description": "Whether to also report code change deltas."}
        },
        "required": ["topic"]
    }

    def __init__(self, rag_instance: Optional[PersistentRAG] = None, **kwargs):
        super().__init__(**kwargs)
        self.rag = rag_instance if rag_instance else PersistentRAG()
        self._file_snapshots = {} # Path -> Content

    def execute(self, topic: str, track_code: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            # 1. Research Iteration Tracking
            all_results = self.rag.query(topic, top_k=10)
            topic_slug = topic.lower().replace(" ", "_")
            topic_entries = [res for res in all_results if topic_slug in res["id"].lower() or topic.lower() in res["text"].lower()[:100]]
            
            research_delta = {}
            if len(topic_entries) >= 2:
                topic_entries.sort(key=lambda x: x["id"], reverse=True)
                current = topic_entries[0]
                previous = topic_entries[1]

                def extract_score(text: str) -> Optional[float]:
                    import re
                    match = re.search(r"Score:\s*([\d\.]+)/(\d+)", text)
                    return float(match.group(1)) if match else None

                curr_score = extract_score(current["text"])
                prev_score = extract_score(previous["text"])
                research_delta = {
                    "latest_score": curr_score,
                    "previous_score": prev_score,
                    "score_delta": round(curr_score - prev_score, 2) if (curr_score is not None and prev_score is not None) else "N/A"
                }

            # 2. Code Change Tracking (Developer-Grade)
            code_delta = []
            if track_code:
                # We check common source files for changes
                for root, dirs, files in os.walk("."):
                    if any(d in root for d in ["__pycache__", ".git", ".selfresearch", "models"]): continue
                    for f in files:
                        if f.endswith((".py", ".json", ".md")):
                            path = os.path.join(root, f)
                            try:
                                with open(path, "r", encoding="utf-8") as f_in:
                                    current_content = f_in.read()
                                
                                if path in self._file_snapshots:
                                    old_content = self._file_snapshots[path]
                                    if old_content != current_content:
                                        diff = list(difflib.unified_diff(
                                            old_content.splitlines(),
                                            current_content.splitlines(),
                                            fromfile=f"prev/{f}",
                                            tofile=f"curr/{f}"
                                        ))
                                        code_delta.append({
                                            "file": path,
                                            "change_summary": f"{len(diff)} lines of diff generated.",
                                            "diff_preview": "".join(diff[:10]) + ("\n..." if len(diff) > 10 else "")
                                        })
                                
                                # Update snapshot for next loop
                                self._file_snapshots[path] = current_content
                            except: pass

            res = {
                "topic": topic,
                "research_metrics": research_delta,
                "code_deltas": code_delta,
                "status": "success"
            }
            log.info(f"IterationTracker: Analyzed '{topic}'. Detected {len(code_delta)} changed files.")
            return res

        except Exception as e:
            log.error(f"Iteration Tracker Error: {e}")
            return {"status": "error", "reason": str(e)}
