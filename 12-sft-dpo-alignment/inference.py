"""DPO 推理：加载 policy 与冻结 reference，比对对数偏好优势。"""
import torch
from project import PreferencePolicy
from utils.v2io import load_torch_state

def infer(directory, text=None):
    prompt = int(text or "0")
    if prompt not in (0,1):
        raise ValueError("示例二分类偏好任务只接受 prompt 0/1")
    policy,reference = PreferencePolicy(),PreferencePolicy()
    load_torch_state(directory,policy)
    load_torch_state(directory,reference,filename="reference.pt")
    with torch.no_grad():
        logp = policy(torch.tensor([prompt]))[0]
        refp = reference(torch.tensor([prompt]))[0]
    chosen=int(logp.argmax().item())
    rejected=1-chosen
    # 与单独的 SFT 概率不同，DPO 比较 policy/ref 两组 log-prob 差。
    delta=(logp[chosen]-logp[rejected])-(refp[chosen]-refp[rejected])
    return {"prompt":prompt,"chosen":chosen,
            "policy_probabilities":logp.exp().tolist(),
            "reference_probabilities":refp.exp().tolist(),
            "relative_log_margin":float(delta)}
