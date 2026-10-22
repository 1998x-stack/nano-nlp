"""Seq2Seq 自回归推理；严禁读取真实 target 作为 decoder 输入。"""
import torch
from model import TinyTranslator
from project import greedy_decode
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    source = [int(s) for s in (text or "1,2,3").replace(" ", ",").split(",") if s]
    if not source or any(s not in (1,2,3) for s in source):
        raise ValueError("toy 数字翻译任务的源 token 仅允许 1,2,3")
    meta = read_torch_metadata(directory)
    net = TinyTranslator(vocab=meta["vocab"], dim=meta["dim"])
    load_torch_state(directory, net)
    with torch.no_grad():
        prediction, weights = greedy_decode(net, torch.tensor([source]), length=len(source))
    return {"source": source, "generated": prediction[0].tolist(),
            "cross_attention": weights[0].tolist()}
