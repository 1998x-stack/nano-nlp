"""可验证推理：从 checkpoint 恢复动作策略，用确定性执行器评分。"""
import torch
from model import OpPolicy,verify
from utils.v2io import load_torch_state

def infer(directory, text=None):
    parts=(text or "add,3,4").replace(" ","").split(",")
    if len(parts)!=3 or parts[0] not in ("add","multiply"):
        raise ValueError("请输入 'add,3,4' 或 'multiply,3,4'")
    op=0 if parts[0]=="add" else 1
    a,b=map(int,parts[1:])
    network=OpPolicy()
    load_torch_state(directory,network)
    with torch.no_grad():
        probs=network.logits[op].softmax(-1)
        action=int(probs.argmax())
    prediction=a+b if action==0 else a*b
    return {"operation":parts[0],"input":[a,b],"prediction":prediction,
            "verified":bool(verify(op,a,b,prediction)),"p_add":float(probs[0])}
