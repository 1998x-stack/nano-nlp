"""09-bert-mlm: v2 完整闭环 CLI（训练、验证、推理、落盘）。"""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for path in (HERE,ROOT):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
if __name__=="__main__":
    import project
    from utils.runtime import run_cli
    run_cli(project,default_steps=24,default_output=str(ROOT/"outputs"/HERE.name))
