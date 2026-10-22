"""v3 manifest：记录数据 hash，检查损坏和跨进程 inspect。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from utils.provenance import sha256_file, verify_manifest

ROOT = Path(__file__).resolve().parents[1]

class ProvenanceTests(unittest.TestCase):
    def test_manifest_is_verifiable_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = [sys.executable, str(ROOT/"02-ngram-language-model"/"demo.py")]
            run = subprocess.run(runner+["--output-dir",directory],
                     cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stderr)
            manifest = verify_manifest(directory)
            self.assertEqual(manifest["schema_version"],3)
            self.assertEqual(manifest["chapter"],"02-ngram-language-model")
            self.assertEqual(manifest["source"]["kind"],"builtin_toy")
            for name,item in manifest["artifacts"].items():
                self.assertEqual(sha256_file(Path(directory)/name),item["sha256"])
            inspection = subprocess.run(runner+["--mode","inspect","--output-dir",directory],
                           cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(inspection.returncode,0,inspection.stderr)
            result=json.loads(inspection.stdout.split("RESULT:",1)[1].strip())
            self.assertTrue(result["verified"])
            with (Path(directory)/"checkpoint.json").open("a",encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaises(ValueError):
                verify_manifest(directory)
            blocked = subprocess.run(runner+["--mode","predict","--output-dir",directory],
                           cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(blocked.returncode,0)

    def test_manifest_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.txt"
            path.write_text("original",encoding="utf-8")
            first=sha256_file(path)
            path.write_text("modified",encoding="utf-8")
            self.assertNotEqual(first,sha256_file(path))
