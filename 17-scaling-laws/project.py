"""Scaling Laws 玩具预算规划器：搜索参数/token 网格、验证预算、对比决策。"""
from loguru import logger
from data import PARAMS,TOKENS,BUDGETS
from model import synthetic_loss

def optimal(budget):
    candidates=[(synthetic_loss(n,d),n,d) for n in PARAMS for d in TOKENS if n*d<=budget]
    if not candidates:
        raise ValueError("预算太小，无可行候选")
    return min(candidates,key=lambda x:x[0]),len(candidates)

def run(*,steps=1,seed=42,text_file=None):
    rows=[]
    for budget in BUDGETS:
        (loss,n,d),num=optimal(budget)
        assert n*d<=budget
        rows.append({"budget":budget,"params_units":n,"tokens_units":d,
                     "estimated_loss":round(loss,6),"feasible_candidates":num,
                     "compute_used":n*d})
        logger.info("budget={} optimum N={} D={} toy_loss={:.5f}",budget,n,d,loss)
    # 实验性规律：扩大预算不应使“最优”损失变差。
    assert all(rows[i]["estimated_loss"]<=rows[i-1]["estimated_loss"]+1e-6
               for i in range(1,len(rows)))
    return {"task":"compute_allocation","scenario":"synthetic_coefficients_NOT_empirical",
            "num_budgets":len(rows),"best_loss":rows[-1]["estimated_loss"],
            "largest_budget":rows[-1]["budget"]},rows
