"""数据格式测试：JSONL 文本拼接不能把 source/field 元信息混进语言模型。"""
import json
import tempfile
import unittest
from pathlib import Path
from utils.data import read_text, nonoverlap_split

class DataContracts(unittest.TestCase):
    def test_jsonl_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"samples.jsonl"
            with p.open("w",encoding="utf-8") as f:
                for s in ["learn language models", "attention connects tokens"]:
                    f.write(json.dumps({"text":s,"source":"should_NOT_be_training_token"})+"\n")
            content=read_text(p)
            self.assertIn("attention connects tokens",content)
            self.assertNotIn("should_NOT_be_training_token",content)

    def test_jsonl_must_have_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"invalid.jsonl"
            p.write_text('{"context":"not text"}\n',encoding="utf-8")
            with self.assertRaises(ValueError):
                read_text(p)

    def test_disjoint_split_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            nonoverlap_split([{"x":1},{"x":1}])
