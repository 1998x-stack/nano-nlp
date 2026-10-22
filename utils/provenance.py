"""v3 实验血缘与产物完整性。

manifest 不是数字签名，不能证明 checkpoint 来自可信作者；它解决
*误用旧工件 / 文件意外改动 / 无法识别数据版本* 这些可复现性问题。
只计算本地文件的 SHA-256，不访问网络或发送文本。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

SCHEMA = 3
ARTIFACTS = ("metrics.json", "examples.jsonl", "history.jsonl",
             "checkpoint.json", "checkpoint.pt", "reference.pt")

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()

def _git_revision(root: Path):
    """仓库未以 git checkout 形式提供时返回 None，不影响离线运行。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def write_manifest(directory, *, project_file, task, seed, steps, text_file):
    """metrics/examples/history 与 checkpoint 全部写完后才生成 manifest。"""
    output = Path(directory)
    root = Path(__file__).resolve().parents[1]
    chapter = Path(project_file).resolve().parent.name
    source_path = Path(text_file).resolve() if text_file else root / chapter / "data.py"
    source = {
        "kind": "user_file" if text_file else "builtin_toy",
        "name": source_path.name,         # 避免记录用户完整绝对路径
        "sha256": sha256_file(source_path),
    }
    artifacts = {
        name: {"sha256": sha256_file(output / name),
               "bytes": (output / name).stat().st_size}
        for name in ARTIFACTS if (output / name).is_file()
    }
    if "metrics.json" not in artifacts or "examples.jsonl" not in artifacts:
        raise ValueError("缺少基本评估报告，不能建立运行血缘")
    if not ("checkpoint.pt" in artifacts or "checkpoint.json" in artifacts):
        raise ValueError("缺少 checkpoint，不能建立完整训练血缘")

    payload = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": "nano-nlp",
        "chapter": chapter,
        "task": task,
        "seed": int(seed),
        "steps": int(steps),
        "code_revision": _git_revision(root),
        "source": source,
        "artifacts": artifacts,
        "evaluation_scope": "toy_or_local_split_NOT_official_benchmark",
        "note": "SHA-256 detects accidental changes but is not a security signature",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def verify_manifest(directory):
    """检查文件未被意外修改；不存在或不匹配则拒绝认作已验证运行。"""
    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA or payload.get("project") != "nano-nlp":
        raise ValueError("manifest 不兼容")
    recorded = payload.get("artifacts")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("manifest 没有产物 hash")
    for filename, entry in recorded.items():
        if filename not in ARTIFACTS or not isinstance(entry, dict):
            raise ValueError("manifest 包含未知或无效产物类型")
        path = output / filename
        if not path.is_file() or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"工件丢失或文件大小被改动: {filename}")
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"工件 SHA-256 校验失败: {filename}")
    return payload
