"""17 章 train→persist→exit→restore→predict 端到端验收。

测试使用真正独立的两个 Python 进程，避免依赖 project.run 的局部变量。
默认 CPU 和离线数据，不需要下载论文、数据集或大模型。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name[:2].isdigit())

class RestoreTests(unittest.TestCase):
    def test_all_chapters_restore_in_new_process(self):
        self.assertEqual(len(LESSONS), 17)
        for chapter in LESSONS:
            with self.subTest(chapter=chapter.name), tempfile.TemporaryDirectory() as directory:
                exe = [sys.executable, str(chapter/"demo.py")]
                train = subprocess.run(exe+["--steps","2","--output-dir",directory],
                                       capture_output=True,text=True,cwd=ROOT,timeout=120)
                self.assertEqual(train.returncode,0,train.stderr)
                self.assertIn("RESULT:",train.stdout)
                checkpoint_json = Path(directory)/"checkpoint.json"
                checkpoint_pt = Path(directory)/"checkpoint.pt"
                self.assertTrue(checkpoint_json.is_file() or checkpoint_pt.is_file(),
                                f"{chapter.name} 没有保存可恢复的模型状态")
                self.assertTrue((Path(directory)/"history.jsonl").is_file())
                metrics = json.loads((Path(directory)/"metrics.json").read_text(encoding="utf-8"))
                self.assertIn("task",metrics["metrics"])
                predict = subprocess.run(exe+["--mode","predict","--output-dir",directory],
                                         capture_output=True,text=True,cwd=ROOT,timeout=120)
                self.assertEqual(predict.returncode,0,predict.stderr)
                self.assertIn("RESULT:",predict.stdout)
                parsed=json.loads(predict.stdout.split("RESULT:",1)[1].strip())
                self.assertTrue(parsed)
                self.assertIsInstance(parsed,dict)
                # 无监督字符 LM 的真实新 token 长度，以及完整模型恢复。
                if chapter.name=="10-gpt-causal-lm":
                    self.assertTrue(parsed["checkpoint_reloaded"])
                    self.assertEqual(parsed["new_tokens"],40)
                if chapter.name=="13-rag-retrieval":
                    self.assertTrue(all(ref in {d["id"] for d in
                        json.loads(checkpoint_json.read_text())["state"]["documents"]}
                        for ref in parsed["references"]))
                if chapter.name=="16-lora-finetuning":
                    self.assertEqual(len(parsed["adapter_delta"]),4)
                if chapter.name=="15-data-cleaning-minhash":
                    self.assertIn("is_near_duplicate",parsed)

    def test_illegal_inputs_have_explicit_error(self):
        samples = {
            "02-ngram-language-model":"",  # 非空的未知词并非错误，应由平滑处理
            "07-seq2seq-attention":"99,1,2",
            "09-bert-mlm":"1,2,3,4",         # 缺少 [MASK]
            "10-gpt-causal-lm":"☃",          # 未见字符
            "16-lora-finetuning":"1,2",      # 维度错误
        }
        for name,input_text in samples.items():
            if not input_text:
                continue
            with self.subTest(chapter=name), tempfile.TemporaryDirectory() as directory:
                exe=[sys.executable,str(ROOT/name/"demo.py")]
                proc=subprocess.run(exe+["--steps","2","--output-dir",directory],
                                    text=True,capture_output=True,cwd=ROOT,timeout=120)
                self.assertEqual(proc.returncode,0,proc.stderr)
                bad=subprocess.run(exe+["--mode","predict","--output-dir",directory,
                                          "--input",input_text],
                                   text=True,capture_output=True,cwd=ROOT,timeout=120)
                self.assertNotEqual(bad.returncode,0)
                self.assertTrue("ValueError" in bad.stderr or "Error" in bad.stderr)

    def test_training_history_is_measured_not_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            proc=subprocess.run(
                [sys.executable,str(ROOT/"10-gpt-causal-lm"/"demo.py"),
                 "--steps","3","--output-dir",directory],
                cwd=ROOT,text=True,capture_output=True,timeout=120)
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows=[json.loads(s) for s in
                  (Path(directory)/"history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows],[1,2,3])
            self.assertTrue(all(isinstance(row["loss"],float) for row in rows))
            self.assertTrue(all(row["stage"]=="train" for row in rows))

if __name__=="__main__":
    unittest.main()
