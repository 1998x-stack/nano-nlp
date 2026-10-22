"""12-sft-dpo-alignment: 完整 MVP 实验入口。"""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for path in (HERE,ROOT):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
if __name__!="__main__" and HERE.name in ("15-data-cleaning-minhash","17-scaling-laws"):
    import importlib.util
    name="_nanonlp_"+HERE.name.replace("-","_")
    spec=importlib.util.spec_from_file_location(name,HERE/"model.py")
    model=importlib.util.module_from_spec(spec);sys.modules[name]=model;spec.loader.exec_module(model)
    for key in (("minhash","deduplicate","DOCS") if HERE.name=="15-data-cleaning-minhash" else ("search_budget","synthetic_loss")):
        globals()[key]=getattr(model,key)
if __name__=="__main__":
    import project
    from utils.runtime import run_cli
    run_cli(project,default_steps=30,default_output=str(ROOT/"outputs"/HERE.name))
