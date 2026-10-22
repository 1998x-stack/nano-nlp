"""加载 AttentionClassifier，观察分类概率和因果多头注意力权重。"""
import torch
from project import AttentionClassifier
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    values = [int(s) for s in (text or "1,2,3,0").replace(" ", ",").split(",") if s]
    meta = read_torch_metadata(directory)
    if not 1 <= len(values) <= meta["max_length"] or any(not 0 <= s < meta["vocab"] for s in values):
        raise ValueError("输入序列必须为 1–4 个 0..3 范围内的整数")
    net = AttentionClassifier()
    load_torch_state(directory, net)
    with torch.no_grad():
        logits, attention = net(torch.tensor([values]))
    return {"tokens": values, "predicted_class": int(logits.argmax(-1)),
            "probabilities": logits.softmax(-1)[0].tolist(),
            "attention_heads": attention[0].tolist()}
