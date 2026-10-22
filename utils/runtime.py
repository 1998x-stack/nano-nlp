"""实验运行时：统一 seed、loguru 和可检查工件。标准日志只在 CLI 配置。"""
import json
import math
import random
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]

def seed_everything(seed: int) -> None:
    """固定 Python 和可选 PyTorch 的随机性；不承诺跨硬件严格一致。"""
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def configure_logging(directory: str | Path, level: str = "INFO") -> Path:
    """日志只配置一次，终端简洁，日志文件可用于复现。"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}")
    logfile = path / "run.log"
    logger.add(str(logfile), level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{line} | {message}",
               rotation="5 MB", retention=2, encoding="utf-8")
    return path

def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metrics.json 不允许 NaN/Infinity")
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value

def save_metrics(directory, metrics: dict, *, seed=42, config=None):
    """将训练和验证结果保留为可检查的稳定 JSON。"""
    result = {"seed": seed, "config": config or {}, "metrics": _jsonable(metrics)}
    path = Path(directory) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path

def save_examples(directory, examples):
    """每行一个示例，方便外部复现、分析及错误切片。"""
    path = Path(directory) / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for example in examples:
            out.write(json.dumps(_jsonable(example), ensure_ascii=False, allow_nan=False) + "\n")
    return path

def run_cli(project, *, default_steps=30, default_output=None):
    """CLI 委托给本章 project.run，保持教学代码不隐藏在 framework 内。"""
    import argparse
    parser = argparse.ArgumentParser(description=project.__doc__ or "NLP mini project")
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=default_output or str(ROOT / "outputs" / Path(project.__file__).resolve().parent.name))
    parser.add_argument("--text-file", type=Path, help="可选本地数据文本；仅支持该选项的章节会读取它")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps 必须 >= 1")
    target = configure_logging(args.output_dir)
    seed_everything(args.seed)
    logger.info("starting experiment: {} | seed={} | steps={} | output={}",
                project.__file__, args.seed, args.steps, target)
    metrics, examples = project.run(steps=args.steps, seed=args.seed, text_file=args.text_file)
    save_metrics(target, metrics, seed=args.seed, config={"steps":args.steps, "text_file":str(args.text_file) if args.text_file else None})
    save_examples(target, examples)
    logger.info("experiment done | metrics={} | examples={}", metrics, len(examples))
    # 只有 stdout 的 RESULT 标识为机器可读 summary；loguru 输出走 stderr。
    print("RESULT:", json.dumps(_jsonable(metrics), ensure_ascii=False, allow_nan=False))
