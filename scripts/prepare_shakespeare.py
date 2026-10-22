"""取 nanoGPT 使用的 Tiny Shakespeare 文本，独立保存；非运行测试必须步骤。"""
import argparse
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).resolve().parents[1]

def prepare(limit_chars=50000):
    path = ROOT / "data" / "tiny_shakespeare.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=30) as response:
            data = response.read().decode("utf-8")
        path.write_text(data, encoding="utf-8")
    text = path.read_text(encoding="utf-8")[:limit_chars]
    target = ROOT / "data" / "tiny_shakespeare_sample.txt"
    target.write_text(text, encoding="utf-8")
    return target

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit-chars", type=int, default=50000)
    args = p.parse_args()
    print("RESULT:", prepare(args.limit_chars))
