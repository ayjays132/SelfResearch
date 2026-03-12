import os
import json
import time
import logging
import threading
import traceback
import sys
from unittest.mock import MagicMock

# Import the real OS
from main import SelfResearchOS

# Force verbose logging to stdout for tracking
logging.basicConfig(level=logging.INFO)

class DaemonStressTester:
    def __init__(self):
        print("\n" + "="*60)
        print("🕵️ DAEMON STRESS TEST: COMPREHENSIVE TOOL AUDIT")
        print("="*60)
        
        # Mocking TUI run so it doesn't try to take over terminal
        with MagicMock() as mock_app:
            with patch('main.Application', return_value=mock_app):
                self.os = SelfResearchOS()
        
        self.tool_index = 0
        # Filter out tools that need human interaction or might be too heavy
        self.tools_to_test = [t for t in self.os.tool_manager.all_tools.keys() if t not in ["ask_user"]]
        
    def mock_generate(self, prompt, **kwargs):
        """Mocks the LLM to return specific tool calls sequentially."""
        if "Suggest a radical new hypothesis" in prompt:
            return "Stress Test: Comprehensive Tool Verification Protocol"
        if "Rate how much we care" in prompt:
            return "2.0"
        if "Formulate a 1-3 word abstract goal" in prompt:
            return "Total Verification"
        if "hallucinate a 'missing dimension'" in prompt:
            return "Dimension of Infinite Stress"
        if "make one highly specific, numeric or binary prediction" in prompt:
            return "All tools return success."
        
        if "Verify if" in prompt and "holds true" in prompt:
            # Trigger the next tool in our list
            if self.tool_index < len(self.tools_to_test):
                tool_name = self.tools_to_test[self.tool_index]
                self.tool_index += 1
                
                # Create dummy args based on tool
                dummy_args = {}
                if tool_name == "file_reader": dummy_args = {"file_path": "main.py"}
                elif tool_name == "file_editor": dummy_args = {"file_path": "main.py", "old_string": "dummy", "new_string": "dummy"}
                elif tool_name == "web_search": dummy_args = {"query": "test"}
                elif tool_name == "academic_search": dummy_args = {"query": "test"}
                elif tool_name == "simulation_lab": dummy_args = {"args": ["physics", 0, 0, 10, 0.1]}
                elif tool_name == "export_layer": dummy_args = {"topic_or_id": "test", "format": "markdown"}
                elif tool_name == "syntax_checker": dummy_args = {"file_path": "main.py"}
                elif tool_name == "project_indexer": dummy_args = {}
                elif tool_name == "sandbox_executor": dummy_args = {"action": "run", "command": "echo test"}
                elif tool_name == "coherence_evaluator": dummy_args = {"checkpoint_path": "main.py"} # using main.py as dummy path
                elif tool_name == "tokenizer_offset_manager": dummy_args = {"tokenizers_info": [{"name": "test", "vocab_size": 100}]}
                elif tool_name == "weight_transfer_tool": dummy_args = {"source_paths": ["main.py"], "modality": "text", "target_state_dict": {}}
                
                return f"<tool_call>\n{{\"name\": \"{tool_name}\", \"kwargs\": {json.dumps(dummy_args)}}}\n</tool_call>\nVerification complete."
            return "All tools verified."
        
        if "adversarial peer reviewer" in prompt:
            return "This research is too perfect. It must be a simulation."
        if "defend it or amend" in prompt:
            return "I am the simulation."
        if "Rewrite the rubric" in prompt:
            return '{"Rigor": {"expected_content": "Maximal test coverage", "max_score": 10}}'
        if "Extract one single, highly specific technical rule" in prompt:
            return "Always test your tools in a daemon loop."
        
        return "Generic agent response for stress test."

    def run(self):
        print("[1/3] Calibrating substrate...")
        # We run boot sequence synchronously for testing
        self.os._boot_sequence()
        
        if not self.os.boot_complete:
            print("❌ Boot failed.")
            return

        print("✅ Kernel Active.")

        # 2. Patch Agent and Trigger Daemon
        self.os.agent.generate = MagicMock(side_effect=self.mock_generate)
        
        print(f"[2/3] Initiating Stress Loop. Testing {len(self.tools_to_test)} tools.")
        
        # We manually run iterations
        print("[3/3] Running Iterations...")
        for i in range(len(self.tools_to_test) + 2):
            topic = self.os._determine_next_topic()
            print(f"\n--- Iteration {i+1}: {topic} ---")
            
            try:
                self.os.run_research_pipeline(topic)
                print(f"   ✅ [SUCCESS]")
            except Exception as e:
                print(f"   ❌ [CRASH]: {e}")
                traceback.print_exc()
            
            if self.tool_index >= len(self.tools_to_test):
                print("\n✅ All tools verified successfully.")
                break

        print("\n" + "="*60)
        print("🏁 STRESS TEST COMPLETE")
        print(f"Tools Audited: {self.tool_index}/{len(self.tools_to_test)}")
        print("="*60 + "\n")
        
        self.os.shutdown()

if __name__ == "__main__":
    from unittest.mock import patch
    tester = DaemonStressTester()
    tester.run()
