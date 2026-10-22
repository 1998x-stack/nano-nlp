"""标准库数据入口：不在模型模块中访问网络。"""
import json
from pathlib import Path

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows

def read_text(path, *, min_length=20):
    value = Path(path).read_text(encoding="utf-8")
    if len(value) < min_length:
        raise ValueError(f"文本至少需要 {min_length} 个字符，实际 {len(value)}")
    return value

def split_items(items, validation_ratio=.25):
    """固定顺序的小数据划分，避免测试数据混入拟合。"""
    if len(items) < 2:
        raise ValueError("需至少 2 条样本")
    n = max(1, min(len(items)-1, round(len(items)*validation_ratio)))
    return list(items[:-n]), list(items[-n:])
