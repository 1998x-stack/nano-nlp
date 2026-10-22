"""离线 smoke + 核心数学不变量；不依赖外部网络和 Hugging Face。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
LESSONS = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())

def load_module(folder):
    path = ROOT / folder / "demo.py"
    name = "demo_" + folder.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class CurriculumTests(unittest.TestCase):
    def test_structure(self):
        self.assertEqual(len(LESSONS), 17)
        for folder in LESSONS:
            for filename in ("README.md", "demo.py", "index.html"):
                with self.subTest(folder=folder.name, filename=filename):
                    self.assertTrue((folder / filename).is_file())
                    self.assertGreater((folder / filename).stat().st_size, 100)

    def test_dataset_mapping(self):
        catalog = json.loads((ROOT / "datasets.json").read_text())
        self.assertEqual({x.name for x in LESSONS}, set(catalog))
        for spec in catalog.values():
            self.assertTrue(spec["url"].startswith("https://huggingface.co/datasets/"))

    def test_cfg(self):
        m = load_module("01-symbolic-grammar")
        self.assertTrue(m.parse("the cat sees a dog"))
        self.assertFalse(m.parse("the sees cat"))

    def test_bigram_normalization(self):
        m = load_module("02-ngram-language-model")
        model = m.BigramLM().fit(m.CORPUS)
        self.assertAlmostEqual(sum(x[1] for x in model.predict("likes")), 1.0)
        self.assertGreater(model.probability("likes","fish"), model.probability("likes","dog"))

    def test_viterbi(self):
        m = load_module("03-hmm-viterbi")
        path, score = m.viterbi(["dogs", "run"])
        self.assertEqual(path, ["N","V"])
        self.assertLess(score,0)

    def test_bm25(self):
        m = load_module("04-tfidf-bm25-search")
        search = m.BM25(m.DOCS)
        self.assertEqual(search.search("attention language")[0],0)
        self.assertEqual(search.score("absentword",0),0)

    def test_shared_retrieval(self):
        from utils.retrieval import BM25
        docs = ["attention networks", "language statistics"]
        self.assertEqual(BM25(docs).search("attention")[0],0)
        self.assertEqual(BM25(docs).score("unknown",0),0)

    def test_rag_traceability(self):
        m = load_module("13-rag-retrieval")
        prompt = m.build_prompt("Transformer",m.retrieve("Transformer",k=1))
        self.assertIn("[D1]",prompt)
        self.assertIn("证据:",prompt)

    def test_minhash_deterministic(self):
        m = load_module("15-data-cleaning-minhash")
        self.assertEqual(m.minhash("The CAT!"),m.minhash("the cat"))
        self.assertLess(len(m.deduplicate(m.DOCS)),len(m.DOCS))

    def test_scaling_budget(self):
        m = load_module("17-scaling-laws")
        best, _ = m.search_budget(1024)
        self.assertLessEqual(best[1]*best[2],1024)


    def test_local_text_input(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安装")
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.txt"
            path.write_text("simple text for tiny language models! " * 4, encoding="utf-8")
            for lesson in ("06-rnn-lstm", "10-gpt-causal-lm"):
                with self.subTest(lesson=lesson):
                    proc = subprocess.run(
                        [sys.executable, str(ROOT/lesson/"demo.py"),
                         "--steps", "2", "--text-file", str(path)],
                        cwd=ROOT, capture_output=True, text=True, timeout=50)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertIn("RESULT:", proc.stdout)

    def test_optional_torch_smoke(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安装；纯 Python 测试仍有效")
        for lesson in ("05-word2vec-skipgram","06-rnn-lstm",
                       "07-seq2seq-attention","08-transformer-attention",
                       "09-bert-mlm","10-gpt-causal-lm",
                       "11-t5-span-corruption","12-sft-dpo-alignment",
                       "14-reasoning-rl-verifier","16-lora-finetuning"):
            with self.subTest(lesson=lesson):
                command = [sys.executable, str(ROOT/lesson/"demo.py")]
                if lesson != "08-transformer-attention":
                    command += ["--steps","2"]
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=50, cwd=ROOT)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn("RESULT:",result.stdout)

if __name__ == "__main__":
    unittest.main()
