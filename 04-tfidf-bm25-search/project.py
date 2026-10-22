"""BM25 搜索引擎 MVP：索引→top-k→MRR/Recall@k→输出评分明细。"""
from loguru import logger
from utils.v2io import save_json_state, load_json_state
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
    save_json_state({"documents": DOCS, "k1": engine.k1, "b": engine.b})

    return {"task":"retrieval","documents":len(DOCS),"queries":len(VALID),
            "mrr":mean(r["mrr"] for r in rows),
            "recall_at_3":mean(r["recall_at_3"] for r in rows)},rows


def predict_saved(directory, text=None):
    """使用保存的文档与 BM25 参数重建倒排统计并检索。"""
    state = load_json_state(directory)
    docs = state["documents"]
    engine = BM25([d["text"] for d in docs], k1=state["k1"], b=state["b"])
    query = text or "attention language"
    hits = engine.search(query)[:3]
    return {"query": query, "hits": [{"id": docs[i]["id"],
            "text": docs[i]["text"], "score": round(engine.score(query, i), 6)} for i in hits]}
