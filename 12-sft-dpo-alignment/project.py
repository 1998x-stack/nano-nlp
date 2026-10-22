"""SFT→固定 reference→DPO→偏好胜率评价。这里只是单 token 教学任务。"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TRAIN, VALID

class PreferencePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(2,12)
        self.head=nn.Linear(12,2)
    def forward(self,prompts):
        return self.head(self.embed(prompts)).log_softmax(-1)

def margin(logp,prompt,chosen,rejected):
    values=logp(prompt)
    return values.gather(1,chosen[:,None]).squeeze(1)-values.gather(1,rejected[:,None]).squeeze(1)

def run(*,steps=30,seed=42,text_file=None):
    """先 SFT 后冻结 reference，DPO 优化 chosen/rejected 相对对数优势。

    核心不变量：冻结的 reference 不产生梯度，chosen 与 rejected 始终属于同一 prompt。
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train,valid=TRAIN,VALID
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        rows=read_jsonl(text_file)
        if len(rows)<4 or not all(type(r.get("prompt")) is int and
                 type(r.get("chosen")) is int and type(r.get("rejected")) is int
                 and r["prompt"] in (0,1) and r["chosen"] in (0,1)
                 and r["rejected"]==1-r["chosen"] for r in rows):
            raise ValueError("DPO JSONL: prompt/chosen/rejected整数0/1; chosen与rejected不同")
        a,b=nonoverlap_split(rows)
        train=[(r["prompt"],r["chosen"],r["rejected"]) for r in a]
        valid=[(r["prompt"],r["chosen"],r["rejected"]) for r in b]
    prompt=torch.tensor([r[0] for r in train])
    chosen=torch.tensor([r[1] for r in train])
    rejected=torch.tensor([r[2] for r in train])
    policy=PreferencePolicy()
    optim=torch.optim.AdamW(policy.parameters(),lr=.07)
    sft_steps=max(1,steps//2)
    for t in range(sft_steps):
        loss=F.nll_loss(policy(prompt),chosen)
        optim.zero_grad()
        loss.backward()
        optim.step()
        record_step(loss=float(loss.item()))
    reference=copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        before=margin(policy,prompt,chosen,rejected).mean().item()
    logger.info("SFT finished | reference frozen | initial margin={:.4f}",before)
    beta=.5
    for t in range(max(1,steps-sft_steps)):
        policy_margin=margin(policy,prompt,chosen,rejected)
        with torch.no_grad():
            ref_margin=margin(reference,prompt,chosen,rejected)
        advantage=policy_margin-ref_margin
        loss=-F.logsigmoid(beta*advantage).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        record_step(loss=float(loss.item()))
    q=torch.tensor([r[0] for r in valid])
    true=torch.tensor([r[1] for r in valid])
    false=torch.tensor([r[2] for r in valid])
    with torch.no_grad():
        logp=policy(q)
        refp=reference(q)
        preference_accuracy=(logp.gather(1,true[:,None])>logp.gather(1,false[:,None])).float().mean().item()
        improvement=(margin(policy,q,true,false)-margin(reference,q,true,false)).mean().item()
    rows=[{"prompt":int(p),"chosen":int(c),"rejected":int(r),
           "policy_chosen_logp":round(float(logp[i,c]),5),
           "policy_rejected_logp":round(float(logp[i,r]),5),
           "reference_chosen_logp":round(float(refp[i,c]),5)}
          for i,(p,c,r) in enumerate(valid)]
    save_torch_state(policy, {"vocab": 2, "dim": 12, "beta": beta})
    save_torch_state(reference, {"vocab": 2, "dim": 12}, filename="reference.pt")

    return {"task":"preference_optimization","sft_steps":sft_steps,"dpo_steps":max(1,steps-sft_steps),
            "reference_frozen":all(not p.requires_grad for p in reference.parameters()),
            "val_chosen_win_rate":preference_accuracy,
            "mean_log_margin_improvement":round(improvement,5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
