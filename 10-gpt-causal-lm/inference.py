"""GPT 独立生成：读取同一 checkpoint 的词表、配置、权重。"""
import torch
from model import TinyGPT, Config
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None):
    meta = read_torch_metadata(directory)
    chars = meta["chars"]
    stoi = {ch:i for i,ch in enumerate(chars)}
    prompt = text if text is not None else chars[0]
    if not prompt or any(ch not in stoi for ch in prompt):
        raise ValueError("prompt 不得为空或包含词表外字符；需与训练语料的字符集合一致")
    config = Config(block_size=meta["block_size"], n_embd=meta["n_embd"],
                    n_head=meta["n_head"])
    model = TinyGPT(len(chars),config)
    load_torch_state(directory, model)
    tokens = [stoi[ch] for ch in prompt]
    with torch.no_grad():
        for _ in range(40):
            # 窗口截断确保位置嵌入和注意力掩码不越界。
            x = torch.tensor(tokens[-config.block_size:])[None,:]
            logits,_ = model(x)
            tokens.append(int(logits[0,-1].argmax()))
    return {"prompt":prompt,"generated":"".join(chars[i] for i in tokens),
            "new_tokens":40,"checkpoint_reloaded":True}
