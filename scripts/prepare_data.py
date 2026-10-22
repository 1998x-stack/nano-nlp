"""按 nanoGPT 的 prepare.py 思路：显式下载、裁剪、保存小数据，不在模型训练时联网。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_catalog():
    return json.loads((ROOT / "datasets.json").read_text(encoding="utf-8"))

def prepare(lesson, limit=200, output=None):
    catalog = load_catalog()
    if lesson not in catalog:
        raise ValueError(f"未知章节: {lesson}; 参见 datasets.json")
    spec = catalog[lesson]
    if spec.get("manual"):
        raise ValueError("本数据集当前依赖旧版自定义 loading script；请根据 dataset card 手动导出安全文本，不自动运行远程脚本。")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("需安装 pip install -r requirements-data.txt") from exc
    path = Path(output) if output else ROOT / "data" / lesson / "samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(spec["repo"], name=spec["config"],
                      split=spec["split"], streaming=True)
    with path.open("w", encoding="utf-8") as file:
        count = 0
        for row in ds:
            value = row[spec["field"]]
            if isinstance(value, dict):  # 翻译数据保留语言键
                value = json.dumps(value, ensure_ascii=False)
            record = {"text": str(value), "source": spec["repo"],
                      "field": spec["field"]}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return path, count

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lesson", required=True, help="如 10-gpt-causal-lm")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--output")
    args = p.parse_args()
    if args.limit < 1 or args.limit > 10000:
        p.error("--limit 必须在 1..10000 范围")
    path, count = prepare(args.lesson, args.limit, args.output)
    print(f"已写入 {count} 条样本到 {path}")
