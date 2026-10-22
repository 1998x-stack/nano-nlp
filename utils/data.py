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
    """支持 TXT 与按行存储 text 字段的 Hugging Face JSONL。
    
    不能把 JSONL 的 source/field 等元信息误当训练 token。
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        if not all(isinstance(r, dict) and isinstance(r.get("text"), str) for r in rows):
            raise ValueError("JSONL 每行必须含字符串 text 字段")
        value = "\n".join(row["text"] for row in rows if row["text"].strip())
    else:
        value = path.read_text(encoding="utf-8")
    if len(value) < min_length:
        raise ValueError(f"文本至少需要 {min_length} 个字符，实际 {len(value)}")
    return value

def split_items(items, validation_ratio=.25):
    """固定顺序的小数据划分，避免测试数据混入拟合。"""
    if len(items) < 2:
        raise ValueError("需至少 2 条样本")
    n = max(1, min(len(items)-1, round(len(items)*validation_ratio)))
    return list(items[:-n]), list(items[-n:])

def typed_records(path, record_types):
    """验证外部小型教学数据的任务类别，防止无监督文本冒充监督标签。"""
    records = read_jsonl(path)
    if not all(isinstance(row, dict) and row.get("type") in record_types for row in records):
        raise ValueError(f"每行须有 type 字段，允许：{sorted(record_types)}")
    return records

def nonoverlap_split(items, validation_ratio=0.25):
    """防止完全重复条目同时落入训练和验证区间。"""
    train,valid=split_items(items,validation_ratio)
    keys=lambda xs: {json.dumps(row,sort_keys=True,ensure_ascii=False) for row in xs}
    if keys(train) & keys(valid):
        raise ValueError("训练和验证集合存在完全相同的样本：请先去重，再执行评估")
    return train,valid
