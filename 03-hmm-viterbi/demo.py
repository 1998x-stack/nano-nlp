"""v2 CLI：数据准备→模型→训练/评估→工件；保留 v1 教程 API 供测试。"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# 兼容 v1 原理测试，但不会在 import 时启动训练。
if __name__ != "__main__":
    import importlib.util
    name = "_nanonlp_" + HERE.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, HERE / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for key in ["viterbi","STATES"]:
        globals()[key] = getattr(module, key)

if __name__ == "__main__":
    import project
    from utils.runtime import run_cli
    run_cli(project, default_steps=1, default_output=str(ROOT / "outputs" / HERE.name))
