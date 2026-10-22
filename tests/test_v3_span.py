"""v3 教学 T5 不能有不可解监督冲突与重复验证样例。"""
import unittest
from utils.data import validate_conditional_pairs, reject_overlapping_pairs

class SpanTaskV3Tests(unittest.TestCase):
    def test_conflicting_target_rejected(self):
        samples = [([1,7,3,4,8],[7,2,8,5,9]), ([1,7,3,4,8],[7,5,8,2,9])]
        with self.assertRaises(ValueError):
            validate_conditional_pairs(samples)

    def test_training_examples_are_not_held_out_examples(self):
        train = [([1,7,3,4,8],[7,2,8,5,9])]
        valid = [([2,7,4,3,8],[7,5,8,2,9])]
        self.assertEqual(validate_conditional_pairs(train+valid),2)
        reject_overlapping_pairs(train,valid)
        with self.assertRaises(ValueError):
            reject_overlapping_pairs(train,train)
