"""v3 Bigram 的概率论回归测试，不依赖假定未见词的‘额外概率’。"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load():
    path = ROOT / "02-ngram-language-model" / "model.py"
    spec = importlib.util.spec_from_file_location("v3_ngram", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class BigramV3Tests(unittest.TestCase):
    def test_all_contexts_sum_to_one(self):
        m = load()
        lm = m.BigramLM().fit(["red cat runs", "blue dog runs"])
        for history in ["red", "cat", "never-seen", m.BOS, m.UNK, m.EOS]:
            with self.subTest(history=history):
                mass = sum(lm.probability(history, token) for token in lm.vocab)
                self.assertAlmostEqual(mass, 1.0, places=12)

    def test_oov_is_single_event_not_an_extra_column(self):
        m = load()
        lm = m.BigramLM().fit(["red cat runs"])
        self.assertIn(m.UNK, lm.vocab)
        self.assertNotIn("unseen", lm.vocab)
        self.assertAlmostEqual(lm.probability("red", "unseen"),
                               lm.probability("red", m.UNK))
        self.assertAlmostEqual(lm.probability("unknown context", "cat"),
                               lm.probability(m.UNK, "cat"))

    def test_prediction_does_not_mutate_counts(self):
        m = load()
        lm = m.BigramLM().fit(["red cat runs"])
        old = set(lm.counts)
        lm.predict("unknown context")
        lm.probability("unseen", "never-seen")
        self.assertEqual(old, set(lm.counts))

    def test_refit_resets_training_counts(self):
        m = load()
        lm = m.BigramLM().fit(["cat runs"]).fit(["dog runs"])
        self.assertNotIn("cat", lm.vocab)
        self.assertIn("dog", lm.vocab)
