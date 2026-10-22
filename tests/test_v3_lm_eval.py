"""v3 LM 评估修复：验证 token 全覆盖、训练词表未见字符不泄漏。"""
import unittest

from utils.lm_eval import encode_train_validation, heldout_token_nll

class LMEvaluationV3Tests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        self.torch = torch

    def test_training_vocab_does_not_see_validation_only_character(self):
        # 前 80% 只有 abc，后 20% 出现雪人字符。
        text = ("abc" * 64) + ("ab☃c" * 12)
        chars, train, valid, oov = encode_train_validation(text, block_size=8)
        self.assertEqual(chars[0], "<unk>")
        self.assertNotIn("☃", chars)
        self.assertGreater(oov, 0)
        self.assertGreater(int((valid == 0).sum()), 0)
        self.assertFalse(bool((train == 0).any()))

    def test_all_validation_targets_are_counted(self):
        class FakeModel:
            training = True
            def eval(self):
                self.training = False
                return self
            def train(self, state=True):
                self.training = state
            def __call__(self, x, y):
                # Weighted mean of uneven windows should equal weighted mean token IDs.
                return None, y.float().mean()
        model = FakeModel()
        ids = self.torch.tensor([1,2,3,4,5,6,7,8,9,10,11], dtype=self.torch.long)
        mean, count, chunks = heldout_token_nll(model, ids, block_size=4)
        self.assertEqual([x["tokens"] for x in chunks], [4,4,2])
        self.assertEqual(count, 10)
        self.assertAlmostEqual(mean, float(ids[1:].float().mean()), places=6)
        self.assertTrue(model.training)

    def test_nll_raises_on_short_validation(self):
        class NoForward:
            training = False
        with self.assertRaises(ValueError):
            heldout_token_nll(NoForward(), self.torch.tensor([1]), block_size=4)
