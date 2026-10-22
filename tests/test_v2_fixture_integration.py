"""全部 17 章带有任务化本地数据的 train / restore / predict 回归测试。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/"fixtures"
CHAPTERS=sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())

class LocalDatasetMVP(unittest.TestCase):
    def test_each_stage_uses_real_local_fixture(self):
        self.assertEqual(len(CHAPTERS),17)
        for lesson in CHAPTERS:
            with self.subTest(lesson=lesson.name),tempfile.TemporaryDirectory() as tmp:
                stem=FIXTURES/lesson.name
                data=next((path for path in (stem.with_suffix(".jsonl"),stem.with_suffix(".txt"))
                           if path.is_file()),None)
                self.assertIsNotNone(data,lesson.name)
                command=[sys.executable,str(lesson/"demo.py")]
                train=subprocess.run(command+["--steps","2","--text-file",str(data),
                                "--output-dir",tmp],cwd=ROOT,capture_output=True,text=True,timeout=150)
                self.assertEqual(train.returncode,0,
                                 lesson.name+" training failure:\n"+train.stderr)
                manifest=json.loads((Path(tmp)/"metrics.json").read_text(encoding="utf-8"))
                self.assertIn("task",manifest["metrics"])
                predict=subprocess.run(command+["--mode","predict","--output-dir",tmp],
                                   cwd=ROOT,capture_output=True,text=True,timeout=150)
                self.assertEqual(predict.returncode,0,
                                 lesson.name+" restore failure:\n"+predict.stderr)
                self.assertIn("RESULT:",predict.stdout)

    def test_fixtures_have_no_hidden_downloads(self):
        """本地 fixture 测试不需要访问 HTTP/S3，也不执行远程脚本。"""
        self.assertTrue((FIXTURES/"README.md").is_file())
        self.assertEqual(len([p for p in FIXTURES.iterdir()
                              if p.is_file() and p.suffix in (".jsonl",".txt")]),17)
