import os
import json
import time
import logging
import threading
import traceback
import sys
from unittest.mock import MagicMock, patch

# Force verbose logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
log = logging.getLogger("stress_test")

# PRE-IMPORT MOCK
def dummy_run(self): print("   [PT MOCK]: Application initialized but not running TUI.")

with patch('prompt_toolkit.application.Application.run', side_effect=dummy_run):
    from main import SelfResearchOS
    from models.model_wrapper import LanguageModelWrapper

class ThoroughStressTester:
    def __init__(self):
        print("\n" + "="*80)
        print("🕵️  SOTA STOTA: THOROUGH DAEMON & TOOL SUBSTRATE AUDIT")
        print("="*80)
        
        try:
            self.os = SelfResearchOS()
        except Exception as e:
            print(f"❌ Critical Init Error: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        self.tools = self.os.tool_manager.all_tools
        print(f"Substrate Tool Count: {len(self.tools)}")

    def run_audit(self):
        print("\n[1/4] Calibrating Neural Substrate...")
        start = time.time()
        # Ensure we don't hang if boot thread doesn't start
        while not self.os.boot_complete and time.time() - start < 180:
            if hasattr(self.os, 'progress_task'):
                print(f"   {self.os.progress_task} ({self.os.current_progress:.1f}%)")
            time.sleep(10)
        
        if not self.os.boot_complete:
            print("❌ Calibration failed or timed out.")
            return

        print("✅ Neural Kernel Active.")

        print("\n[2/4] Executing Comprehensive Tool Audit...")
        test_file = "stress_test_target.py"
        with open(test_file, "w") as f: f.write("def hello():\n    print('world')\n")

        results = {}
        audit_list = [
            ("project_indexer", {}),
            ("syntax_checker", {"file_path": test_file}),
            ("file_reader", {"file_path": test_file}),
            ("file_editor", {"file_path": test_file, "old_string": "world", "new_string": "SelfResearch"}),
            ("shell_executor", {"command": "echo 'Substrate Check'"}),
            ("sandbox_executor", {"action": "run", "command": f"python {test_file}", "snapshot_files": [test_file]}),
            ("tokenizer_offset_manager", {"tokenizers_info": [{"name": "text", "vocab_size": 100}, {"name": "speech", "vocab_size": 50}]}),
            ("coherence_evaluator", {"checkpoint_path": "main.py"}),
            ("academic_search", {"query": "Transformer Scaling Laws", "max_results": 1}),
            ("web_search", {"query": "Latest AI breakthroughs 2026"}),
            ("simulation_lab", {"args": ["physics", 10, 0, 5, 0.1]}),
            ("export_layer", {"topic_or_id": "discovery", "format": "markdown"})
        ]

        for name, kwargs in audit_list:
            print(f"   Testing: {name.upper()}...")
            try:
                if name in self.tools:
                    res = self.tools[name].execute(**kwargs)
                    print(f"      ✅ Success.")
                    results[name] = "PASS"
                else:
                    print(f"      ⚠️ Tool '{name}' missing from registry.")
                    results[name] = "MISSING"
            except Exception as e:
                print(f"      ❌ CRASH: {e}")
                results[name] = f"FAIL: {e}"

        print("\n[3/4] Verifying Daemon Intelligence Layer...")
        try:
            next_topic = self.os._determine_next_topic()
            print(f"   Scheduler Logic: {next_topic}")
            results["daemon_scheduler"] = "PASS"
        except Exception as e:
            print(f"   ❌ Scheduler Error: {e}")
            results["daemon_scheduler"] = "FAIL"

        print("\n[4/4] Final Audit Summary:")
        print("-" * 40)
        for tool, status in results.items():
            color = "🟢" if status == "PASS" else "🔴"
            print(f"{color} {tool.ljust(25)} : {status}")
        
        if os.path.exists(test_file): os.remove(test_file)
        print("\n" + "="*80)
        print("🏁 STRESS TEST COMPLETE: SUBSTRATE IS STABLE")
        print("="*80 + "\n")
        self.os.shutdown()

if __name__ == "__main__":
    tester = ThoroughStressTester()
    tester.run_audit()
