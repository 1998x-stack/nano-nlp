"""GPT 独立生成：读取同一 checkpoint 的词表、配置、权重。"""
import torch
from model import TinyGPT, Config
from utils.v2io import read_torch_metadata, load_torch_state

def infer(directory, text=None, *, temperature=0.0, top_k=0, num_new_tokens=40):
    if temperature < 0 or top_k < 0 or not 1 <= num_new_tokens <= 1000:
        raise ValueError("temperature≥0, top-k≥0, new-tokens 范围 1..1000")
    meta = read_torch_metadata(directory)
    chars = meta["chars"]
    stoi = {ch:i for i,ch in enumerate(chars)}
    prompt = text if text is not None else next(ch for ch in chars if len(ch) == 1)
    if not prompt or any(ch not in stoi for ch in prompt):
        raise ValueError("prompt 不得为空或包含词表外字符；需与训练语料的字符集合一致")
    config = Config(block_size=meta["block_size"], n_embd=meta["n_embd"],
                    n_head=meta["n_head"])
    model = TinyGPT(len(chars),config)
    load_torch_state(directory, model)
    tokens = [stoi[ch] for ch in prompt]
    with torch.no_grad():
        for _ in range(num_new_tokens):
            # 只用可见的最后 block_size 个 token；不会把真实未来字符喂进去。
            x = torch.tensor(tokens[-config.block_size:])[None,:]
            logits,_ = model(x)
            next_logits = logits[0,-1]
            if temperature == 0:
                next_id = int(next_logits.argmax())
            else:
                # Top-k 把不在候选集中的词概率设为 0，温度控制随机性。
                if top_k > 0:
                    threshold = torch.topk(next_logits, min(top_k,len(chars))).values[-1]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1))
            tokens.append(next_id)
    return {"prompt":prompt,"generated":"".join(chars[i] for i in tokens),
            "new_tokens":num_new_tokens,"checkpoint_reloaded":True,
            "temperature":temperature,"top_k":top_k}
