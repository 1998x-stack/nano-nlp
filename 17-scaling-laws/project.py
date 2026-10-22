"""Scaling Laws 玩具预算规划器：搜索参数/token 网格、验证预算、对比决策。"""
from loguru import logger
from utils.v2io import save_json_state, load_json_state
from data import PARAMS,TOKENS,BUDGETS
from model import synthetic_loss

def optimal(budget):
    candidates=[(synthetic_loss(n,d),n,d) for n in PARAMS for d in TOKENS if n*d<=budget]
    if not candidates:
        raise ValueError("预算太小，无可行候选")
    return min(candidates,key=lambda x:x[0]),len(candidates)

def run(*,steps=1,seed=42,text_file=None):
    budgets=BUDGETS
    if text_file:
        from utils.data import read_jsonl
        rows=read_jsonl(text_file)
        if not all(isinstance(row.get("budget"),int) and row["budget"]>0 for row in rows):
            raise ValueError("Scaling JSONL 每行需要正整数 budget")
        budgets=sorted(set(row["budget"] for row in rows))
    rows=[]
    for budget in budgets:
        (loss,n,d),num=optimal(budget)
        assert n*d<=budget
        rows.append({"budget":budget,"params_units":n,"tokens_units":d,
                     "estimated_loss":round(loss,6),"feasible_candidates":num,
                     "compute_used":n*d})
        logger.info("budget={} optimum N={} D={} toy_loss={:.5f}",budget,n,d,loss)
    # 实验性规律：扩大预算不应使“最优”损失变差。
    assert all(rows[i]["estimated_loss"]<=rows[i-1]["estimated_loss"]+1e-6
               for i in range(1,len(rows)))
    save_json_state({"budgets": budgets, "params": PARAMS, "tokens": TOKENS,
                     "scenario": "synthetic_coefficients_NOT_empirical"})

    return {"task":"compute_allocation","scenario":"synthetic_coefficients_NOT_empirical",
            "num_budgets":len(rows),"best_loss":rows[-1]["estimated_loss"],
            "largest_budget":rows[-1]["budget"]},rows


def predict_saved(directory, text=None):
    """预算规划器不涉及梯度学习；恢复并固定教学候选网格。"""
    state = load_json_state(directory)
    budget = int(text or str(state["budgets"][-1]))
    choices = [(synthetic_loss(n,d),n,d) for n in state["params"]
               for d in state["tokens"] if n*d <= budget]
    if not choices:
        raise ValueError("预算无法覆盖任一候选配置")
    loss,n,d = min(choices)
    return {"budget": budget, "n": n, "d": d, "loss": round(loss,6),
            "scenario": state["scenario"], "empirical": False}
