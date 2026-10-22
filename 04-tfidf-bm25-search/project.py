"""BM25 搜索引擎 MVP：索引→top-k→MRR/Recall@k→输出评分明细。"""
from loguru import logger
from data import DOCS, VALID
from model import BM25

def run(*, steps=1, seed=42, text_file=None):
    from utils.metrics import mrr, recall_at_k
    engine=BM25([d["text"] for d in DOCS])
    rows=[]
    for sample in VALID:
        ranked_indices=engine.search(sample["query"])
        ranked=[DOCS[i]["id"] for i in ranked_indices]
        scored=[{"id":DOCS[i]["id"],"score":round(engine.score(sample["query"],i),6)} for i in ranked_indices[:3]]
        rows.append({"query":sample["query"],"relevant":sample["relevant"],
                     "ranking":ranked,"top3":scored,"mrr":mrr(ranked,sample["relevant"]),
                     "recall_at_3":recall_at_k(ranked,sample["relevant"],3)})
        logger.debug("query={} | top={} | mrr={}",sample["query"],ranked[:3],rows[-1]["mrr"])
    from utils.metrics import mean
    return {"task":"retrieval","documents":len(DOCS),"queries":len(VALID),
            "mrr":mean(r["mrr"] for r in rows),
            "recall_at_3":mean(r["recall_at_3"] for r in rows)},rows
