"""GPT 自回归 MVP：训练→验证（无泄漏）→ greedy token generation。"""
import math
import torch
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TEXT, BLOCK
from model import TinyGPT, Config

def run(*,steps=30,seed=42,text_file=None):
    """Decoder-only Transformer 只预测下一 token；验证文本切分必须无未来泄漏。

    核心不变量：输入序列右移监督标签，causal mask 使 prefix logits 不依赖后续字符。
    """
    from utils.data import read_text
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    text=read_text(text_file,min_length=40) if text_file else TEXT
    chars=sorted(set(text))
    stoi={c:i for i,c in enumerate(chars)}
    ids=torch.tensor([stoi[ch] for ch in text])
    cut=int(.8*len(ids))
    train,val=ids[:cut],ids[cut:]
    if min(len(train),len(val)) <= BLOCK+1:
        raise ValueError("语料划分后训练与验证段均须长于 block_size")
    cfg=Config(block_size=BLOCK)
    model=TinyGPT(len(chars),cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.008)
    for t in range(steps):
        j=(t*9)%(len(train)-BLOCK-1)
        _,loss=model(train[j:j+BLOCK][None,:],train[j+1:j+BLOCK+1][None,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record_step(loss=float(loss.item()))
        if t in (0,steps-1):logger.info("GPT train step={} nll={:.4f}",t+1,loss.item())
    model.eval()
    with torch.no_grad():
        v_nll=[]
        for j in range(0,len(val)-BLOCK-1,BLOCK):
            _,loss=model(val[j:j+BLOCK][None,:],val[j+1:j+BLOCK+1][None,:])
            v_nll.append(loss.item())
        tokens=[int(train[0])]
        for i in range(40):
            logits,_=model(torch.tensor(tokens[-BLOCK:])[None,:])
            tokens.append(int(logits[0,-1].argmax()))
    text_out="".join(chars[t] for t in tokens)
    nll=sum(v_nll)/len(v_nll)
    save_torch_state(model, {"chars": chars, "block_size": cfg.block_size, "n_embd": cfg.n_embd, "n_head": cfg.n_head})

    return {"task":"causal_language_model","parameters":sum(x.numel() for x in model.parameters()),
            "train_chars":len(train),"valid_chars":len(val),"vocab":len(chars),
            "val_nll":round(nll,5),"val_perplexity":round(math.exp(nll),5),
            "generated":text_out},[{"prompt":text_out[0],"generated":text_out[1:],
                                   "validation_chunk_nll":round(v,5)} for v in v_nll]


def predict_saved(directory, text=None, *, temperature=0.0, top_k=0, num_new_tokens=40):
    """独立恢复模型后，支持 greedy 与温度/top-k 自回归采样。"""
    from inference import infer
    return infer(directory, text, temperature=temperature, top_k=top_k,
                 num_new_tokens=num_new_tokens)
