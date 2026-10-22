"""可验证组奖励 RL 完整闭环：group sampling、优势、策略更新、held-out verifier。"""
import torch
from torch.distributions import Categorical
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TRAIN,VALID
from model import OpPolicy,verify

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    policy=OpPolicy()
    optim=torch.optim.AdamW(policy.parameters(),lr=.12)
    group_size=8
    rewards_history=[]
    for t in range(steps):
        losses=[];mean_rewards=[]
        for op,a,b in TRAIN:
            dist=policy(op)
            actions=dist.sample((group_size,))
            attempts=[a+b if i.item()==0 else a*b for i in actions]
            rewards=torch.tensor([verify(op,a,b,pred) for pred in attempts])
            advantage=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
            losses.append(-(advantage.detach()*dist.log_prob(actions)).mean())
            mean_rewards.append(rewards.float().mean().item())
        loss=torch.stack(losses).mean()
        optim.zero_grad();loss.backward();optim.step(); record_step(loss=float(loss.item()))
        rewards_history.append(sum(mean_rewards)/len(mean_rewards))
        if t in (0,steps-1):logger.info("RL iteration {} group reward={:.4f}",t+1,rewards_history[-1])
    probs=policy.logits.softmax(-1).detach()
    rows=[]
    for op,a,b in VALID:
        action=int(policy.logits[op].argmax())
        prediction=a+b if action==0 else a*b
        gold=a+b if op==0 else a*b
        rows.append({"operator_task":"add" if op==0 else "multiply",
                     "a":a,"b":b,"pred":prediction,"gold":gold,
                     "correct":bool(verify(op,a,b,prediction)),
                     "p_add":round(float(probs[op,0]),4)})
    save_torch_state(policy, {"task": "toy_operator_policy", "group_size": group_size, "reward_history": rewards_history})

    return {"task":"verifiable_policy_optimization","train_tasks":len(TRAIN),
            "valid_tasks":len(VALID),"val_success_rate":sum(r["correct"] for r in rows)/len(rows),
            "last_group_reward":round(rewards_history[-1],5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
