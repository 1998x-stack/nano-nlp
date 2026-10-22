"""跨章节的轻量可恢复工件协议：保持核心算法留在各章 model.py。

所有读取都使用明确的 JSON 或 torch.load(weights_only=True)。
不从未知来源执行 pickle，绝不把模型反序列化当做沙盒。
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from contextvars import ContextVar

from loguru import logger

_OUTPUT = ContextVar("nano_nlp_output", default=None)
_HISTORY = ContextVar("nano_nlp_history", default=None)
SCHEMA = 2

def start_run(directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _OUTPUT.set(directory)
    _HISTORY.set([])

def artifact_dir() -> Path:
    folder = _OUTPUT.get()
    if folder is None:
        raise RuntimeError("必须通过 demo.py --mode train 进入实验，才能写入工件")
    return folder

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("工件不允许非有限浮点数")
    return value

def save_json_state(state: dict, filename="checkpoint.json"):
    """无需反序列化可执行对象的经典 NLP 工件。"""
    target = artifact_dir() / filename
    data = {"schema": SCHEMA, "state": _jsonable(state)}
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("saved JSON model state: {}", target)
    return target

def load_json_state(directory, filename="checkpoint.json"):
    path = Path(directory) / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or not isinstance(data.get("state"), dict):
        raise ValueError(f"不兼容或损坏的 checkpoint: {path}")
    return data["state"]

def save_torch_state(model, metadata=None, filename="checkpoint.pt"):
    """只持久化 state_dict，不序列化整份 Python 模型对象。"""
    import torch
    path = artifact_dir() / filename
    torch.save({"schema": SCHEMA, "meta": _jsonable(metadata or {}),
                "state_dict": model.state_dict()}, path)
    logger.info("saved model weights: {} | parameters={}",
                path, sum(p.numel() for p in model.parameters()))
    return path

def load_torch_state(directory, model, filename="checkpoint.pt"):
    """明确实例化受信任的本地模型类，再加载纯 tensor 的权重。"""
    import torch
    path = Path(directory) / filename
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"checkpoint schema 不匹配：{path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload["meta"]

def read_torch_metadata(directory, filename="checkpoint.pt"):
    """权重与 tokenizer/vocab 元数据共同保存，避免独立推理映射不一致。"""
    import torch
    payload = torch.load(Path(directory) / filename, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("模型 checkpoint 版本不兼容")
    return payload["meta"]

def record_step(loss=None, *, stage="train", **scalars):
    """只记录真实训练执行的数值；不会自动绘制假想 loss。"""
    history = _HISTORY.get()
    if history is None:
        return
    record = {"step": len(history) + 1, "stage": stage}
    if loss is not None:
        record["loss"] = float(loss)
    record.update({k: float(v) for k,v in scalars.items()})
    if not all(math.isfinite(v) for k,v in record.items() if isinstance(v,float)):
        raise ValueError("训练曲线包含非有限数字")
    history.append(record)

def write_history():
    history = _HISTORY.get()
    path = artifact_dir() / "history.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for step in history or []:
            f.write(json.dumps(step, ensure_ascii=False, allow_nan=False) + "\n")
    logger.info("saved true experimental steps: {}", len(history or []))
    return path
