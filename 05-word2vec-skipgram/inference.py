"""使用冻结的训练词向量做最近邻查询，独立于训练进程。"""
import torch
from torch.nn import functional as F
from model import SkipGram
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    meta = read_torch_metadata(directory)
    words = meta["words"]
    query = (text or ("cat" if "cat" in words else words[0])).strip().lower()
    if query not in words:
        raise ValueError(f"未知词 {query!r}，请使用训练集词表中的词")
    model = SkipGram(len(words), dim=meta["dim"])
    load_torch_state(directory, model)
    with torch.no_grad():
        # cosine 与中心词模长无关；剔除查询词自身才是有意义的 top-k。
        emb = F.normalize(model.center.weight, dim=-1)
        sims = emb @ emb[words.index(query)]
        order = torch.argsort(sims, descending=True).tolist()
    return {"query": query, "nearest": [
        {"word": words[i], "cosine": round(float(sims[i]), 5)}
        for i in order if words[i] != query][:5]}
