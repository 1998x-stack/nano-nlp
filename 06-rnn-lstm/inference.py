"""加载保存的词表与 LSTM state_dict，再进行不读标签的字符生成。"""
import torch
from model import LSTMLM
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    meta = read_torch_metadata(directory)
    chars = meta["chars"]
    stoi = {ch:i for i,ch in enumerate(chars)}
    prompt = text if text is not None else next(ch for ch in chars if len(ch) == 1)
    if not prompt or any(ch not in stoi for ch in prompt):
        raise ValueError("prompt 不得为空或包含不在训练语料中的字符")
    net = LSTMLM(len(chars), dim=meta["dim"])
    load_torch_state(directory, net)
    token_ids = [stoi[c] for c in prompt]
    with torch.no_grad():
        for _ in range(24):
            # LSTM 具有序列状态，但这里为教学起见重放一个短上下文。
            x = torch.tensor(token_ids[-meta["block_size"]:], dtype=torch.long)[None,:]
            logits,_ = net(x)
            token_ids.append(int(logits[0,-1].argmax()))
    return {"prompt": prompt, "generated": "".join(chars[i] for i in token_ids),
            "num_new_characters": 24}
