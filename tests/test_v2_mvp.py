"""v2.0 跨模块集成：验证独立评估、推理、日志及可机器读取工件。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHAPTERS=[
    "01-symbolic-grammar","02-ngram-language-model","03-hmm-viterbi",
    "04-tfidf-bm25-search","05-word2vec-skipgram","06-rnn-lstm",
    "07-seq2seq-attention","08-transformer-attention","09-bert-mlm",
    "10-gpt-causal-lm","11-t5-span-corruption","12-sft-dpo-alignment",
    "13-rag-retrieval","14-reasoning-rl-verifier","15-data-cleaning-minhash",
    "16-lora-finetuning","17-scaling-laws"
]

class MVPTests(unittest.TestCase):
    def test_full_pipeline(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter),tempfile.TemporaryDirectory() as tmp:
                command=[sys.executable,str(ROOT/chapter/"demo.py"),
                         "--steps","2","--output-dir",tmp]
                result=subprocess.run(command,cwd=ROOT,capture_output=True,
                                      text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn("RESULT:",result.stdout)
                artifacts=[Path(tmp)/name for name in ("metrics.json","examples.jsonl","run.log")]
                for path in artifacts:
                    self.assertTrue(path.is_file(),path)
                output=json.loads(artifacts[0].read_text(encoding="utf-8"))
                self.assertIn("metrics",output)
                self.assertIn("task",output["metrics"])
                self.assertIn("seed",output)
                self.assertIsInstance(output["metrics"]["task"],str)
                self.assertGreater(len(output["metrics"]),2)
                self.assertTrue(json.loads(artifacts[1].read_text(encoding="utf-8").splitlines()[0]))
                self.assertGreater(artifacts[2].stat().st_size,0)

    def test_hmm_is_estimated_from_training_data(self):
        import importlib.util
        file=ROOT/"03-hmm-viterbi"/"project.py"
        # 避免全局 sys.path 中交叉覆盖不同章节的 data/model。
        source=file.read_text(encoding="utf-8")
        self.assertIn("self.initial[tags[0]] += 1",source)
        self.assertIn("self.transition[tags[i-1]][tag]+=1",source)
        self.assertIn("self.emission[tag][word]+=1",source)

if __name__=="__main__":
    unittest.main()
