"""T5-style span 重建：通过同一个 encoder–decoder 权重完成无标签生成。"""
import torch
from model import TinyT5
from project import generate
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    tokens = [int(s) for s in (text or "1,7,3,4,8").replace(" ", ",").split(",") if s]
    meta = read_torch_metadata(directory)
    if not tokens or len(tokens)>16 or any(not 0<=s<meta["vocab"] for s in tokens):
        raise ValueError("示例 token ID 须位于 0..9，源序列不超过 16 个 token")
    net = TinyT5(vocab=meta["vocab"], dim=meta["dim"])
    load_torch_state(directory, net)
    with torch.no_grad():
        ids = generate(net, torch.tensor([tokens]), length=5)[0]
    return {"corrupted": tokens,"predicted_span_tokens":ids.tolist()}
