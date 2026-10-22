"""GPT 恢复采样行为：greedy、top-k=1 和无效参数。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"10-gpt-causal-lm"/"demo.py"

class SamplingTests(unittest.TestCase):
    def test_top_one_equals_greedy_from_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=[sys.executable,str(SCRIPT)]
            fit=subprocess.run(prefix+["--steps","3","--output-dir",directory],
                               cwd=ROOT,capture_output=True,text=True,timeout=150)
            self.assertEqual(fit.returncode,0,fit.stderr)
            def predict(extra):
                p=subprocess.run(prefix+["--mode","predict","--output-dir",directory,
                         "--input","t","--new-tokens","12"]+extra,
                         cwd=ROOT,capture_output=True,text=True,timeout=150)
                self.assertEqual(p.returncode,0,p.stderr)
                return json.loads(p.stdout.split("RESULT:",1)[1].strip())
            greedy=predict([])
            top_one=predict(["--temperature","0.9","--top-k","1"])
            self.assertEqual(greedy["generated"],top_one["generated"])
            self.assertEqual(greedy["new_tokens"],12)
            self.assertTrue(greedy["checkpoint_reloaded"])

    def test_negative_temperature_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base=[sys.executable,str(SCRIPT)]
            fit=subprocess.run(base+["--steps","2","--output-dir",directory],
                               cwd=ROOT,capture_output=True,text=True,timeout=150)
            self.assertEqual(fit.returncode,0,fit.stderr)
            bad=subprocess.run(base+["--mode","predict","--output-dir",directory,
                     "--temperature","-0.1"],cwd=ROOT,capture_output=True,text=True,timeout=150)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn("ValueError",bad.stderr)
