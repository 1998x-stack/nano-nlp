"""v3 偏好标签一致性：防止将同一 toy prompt 同时偏好相反的答案。"""
import subprocess
import sys
import tempfile
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class PreferenceV3Tests(unittest.TestCase):
    def test_consistent_fixture_is_explicitly_seen_prompt(self):
        with tempfile.TemporaryDirectory() as out:
            result=subprocess.run([sys.executable,str(ROOT/"12-sft-dpo-alignment"/"demo.py"),
                     "--steps","2","--text-file",str(ROOT/"fixtures"/"12-sft-dpo-alignment.jsonl"),
                     "--output-dir",out],cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((Path(out)/"metrics.json").read_text(encoding="utf-8"))["metrics"]
            self.assertEqual(summary["evaluation_scope"],"seen_prompts")
            self.assertGreater(summary["val_prompt_overlap"],0)

    def test_opposite_chosen_for_same_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            p=Path(out)/"contradictory.jsonl"
            examples=[
               {"prompt":0,"chosen":1,"rejected":0,"id":"one"},
               {"prompt":1,"chosen":0,"rejected":1,"id":"two"},
               {"prompt":0,"chosen":0,"rejected":1,"id":"three"},
               {"prompt":1,"chosen":1,"rejected":0,"id":"four"},
            ]
            p.write_text("".join(json.dumps(x)+"\n" for x in examples),encoding="utf-8")
            result=subprocess.run([sys.executable,str(ROOT/"12-sft-dpo-alignment"/"demo.py"),
                     "--steps","2","--text-file",str(p),"--output-dir",out],
                     cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("ValueError",result.stderr)
