"""Skip-gram 完整小训练：正负样本、优化、验证 loss 和 cosine neighbors。"""
from loguru import logger
from utils.v2io import save_torch_state, record_step
import torch
from torch.nn import functional as F
from data import TEXT, CONTEXT_WINDOW
from model import SkipGram, make_pairs

def run(*,steps=30,seed=42,text_file=None):
    """正例来自词窗，负例来自随机采样；可学习参数是两张 embedding。

    核心不变量：训练与验证需要避免直接重用同一正样本索引；近邻用归一化 cosine。
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tokens=list(TEXT)
    if text_file:
        from utils.data import read_text
        tokens=read_text(text_file).lower().split()
    words,pairs=make_pairs(tokens,window=CONTEXT_WINDOW)
    if len(pairs)<8:
        raise ValueError("语料太短，至少需要 8 条中心-上下文词对")
    # 训练/验证按确定性的索引排列拆分；toy 样本只用于机制演示。
    train=pairs[:int(.8*len(pairs))]
    valid=pairs[int(.8*len(pairs)):]
    network=SkipGram(len(words),dim=16)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.03)
    def negatives_for(batch,k=4):
        # 负例排除真正 positive context，但未按 unigram^0.75 加权。
        negatives=torch.randint(0,len(words),(len(batch),k))
        positive=batch[:,1,None]
        negatives=torch.where(negatives==positive,(negatives+1)%len(words),negatives)
        return negatives
    for step in range(steps):
        neg=negatives_for(train)
        loss=network(train[:,0],train[:,1],neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record_step(loss=float(loss.item()))
        if step in (0,steps-1):
            logger.info("skipgram step={} loss={:.4f}",step+1,loss.item())
    with torch.no_grad():
        val_loss=network(valid[:,0],valid[:,1],negatives_for(valid)).item()
        e=F.normalize(network.center.weight,dim=-1)
        cosine=e @ e.T
        rows=[]
        for word in ("cat","dog","fish"):
            if word not in words:
                continue
            i=words.index(word)
            values, indices=cosine[i].topk(min(4,len(words)))
            neighbors=[{"word":words[j],"cosine":round(float(v),4)}
                       for j,v in zip(indices.tolist(),values.tolist()) if j!=i]
            rows.append({"query":word,"nearest":neighbors})
    save_torch_state(network, {"words": words, "dim": 16, "context_window": CONTEXT_WINDOW})

    return {"task":"embedding","vocabulary":len(words),"pairs_train":len(train),
            "pairs_valid":len(valid),"train_loss":round(float(loss.item()),5),
            "val_negative_sampling_loss":round(float(val_loss),5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
