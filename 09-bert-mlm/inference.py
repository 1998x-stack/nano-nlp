"""Encoder 双向补全，不使用自回归采样。"""
import torch
from model import TinyMLM
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    tokens = [int(s) for s in (text or "1,0,3,4").replace(" ", ",").split(",") if s]
    meta = read_torch_metadata(directory)
    if not tokens or len(tokens)>meta["max_length"] or any(not 0<=s<meta["vocab"] for s in tokens):
        raise ValueError("masked tokens 必须为 0..5，长度不超过 16")
    if meta["mask_id"] not in tokens:
        raise ValueError("至少提供一个 0，表示 [MASK]")
    net = TinyMLM(vocab=meta["vocab"])
    load_torch_state(directory, net)
    with torch.no_grad():
        pred = net(torch.tensor([tokens])).argmax(-1)[0].tolist()
    return {"masked_input":tokens,
            "filled_tokens":[pred[i] if s==0 else s for i,s in enumerate(tokens)],
            "masked_predictions":[{"position":i,"token":pred[i]} for i,s in enumerate(tokens) if s==0]}
