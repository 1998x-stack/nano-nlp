"""PEFT 推理：保存了冻结底座和 adapter，按同一参数恢复前向。"""
import torch
from model import LoRALinear
from utils.v2io import read_torch_metadata,load_torch_state

def infer(directory, text=None):
    meta=read_torch_metadata(directory)
    values=[float(s) for s in (text or "1,0,0,0,0,0,0,0").replace(" ", ",").split(",") if s]
    if len(values)!=meta["in_dim"]:
        raise ValueError(f"需要 {meta['in_dim']} 个浮点数作为输入")
    network=LoRALinear(in_dim=meta["in_dim"],out_dim=meta["out_dim"],
                       rank=meta["rank"],alpha=meta["alpha"])
    load_torch_state(directory,network)
    with torch.no_grad():
        x=torch.tensor([values],dtype=torch.float32)
        y=network(x)[0]
        base=network.base(x)[0]
    return {"input":values,"prediction":y.tolist(),
            "base_only":base.tolist(),"adapter_delta":(y-base).tolist()}
