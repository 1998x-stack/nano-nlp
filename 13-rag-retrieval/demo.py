"""13-rag-retrieval: 完整 MVP 实验入口。"""
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
if __name__!="__main__":
    # 对外保留 v1 的检索与 Prompt API；主程序仍走 v2 项目闭环。
    import importlib.util
    name="_nanonlp_13_rag_original"
    spec=importlib.util.spec_from_file_location(name,HERE/"model.py")
    reference=importlib.util.module_from_spec(spec)
    sys.modules[name]=reference
    spec.loader.exec_module(reference)
    retrieve=reference.retrieve
    build_prompt=reference.build_prompt
    CORPUS=reference.CORPUS
if __name__=="__main__":
    import project
    from utils.runtime import run_cli
    run_cli(project,default_steps=1,default_output=str(ROOT/"outputs"/HERE.name))
