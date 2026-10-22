"""BM25 搜索引擎 MVP：索引→top-k→MRR/Recall@k→输出评分明细。"""
from loguru import logger
from utils.v2io import save_json_state, load_json_state
from data import DOCS, VALID
from model import BM25

def run(*, steps=1, seed=42, text_file=None):
    from utils.metrics import mrr, recall_at_k
    docs,valid=DOCS,VALID
    if text_file:
        from utils.data import typed_records
        rows=typed_records(text_file,{"document","query"})
        docs=[{"id":r["id"],"text":r["text"]} for r in rows if r["type"]=="document"]
        valid=[{"query":r["query"],"relevant":r["relevant"]} for r in rows if r["type"]=="query"]
        ids=[d["id"] for d in docs]
        if not docs or not valid or len(ids)!=len(set(ids)):
            raise ValueError("搜索数据需有唯一 id 的 document 与至少一个 query")
        if not all(isinstance(d["text"],str) and d["text"].strip() for d in docs):
            raise ValueError("document.text 必须非空")
        if not all(isinstance(q["query"],str) and isinstance(q["relevant"],list)
                   and set(q["relevant"]) <= set(ids) for q in valid):
            raise ValueError("query.relevant 必须引用 document.id")
    engine=BM25([d["text"] for d in docs])
    rows=[]
    for sample in valid:
        ranked_indices=engine.search(sample["query"])
        ranked=[docs[i]["id"] for i in ranked_indices]
        scored=[{"id":docs[i]["id"],"score":round(engine.score(sample["query"],i),6)} for i in ranked_indices[:3]]
        rows.append({"query":sample["query"],"relevant":sample["relevant"],
                     "ranking":ranked,"top3":scored,"mrr":mrr(ranked,sample["relevant"]),
                     "recall_at_3":recall_at_k(ranked,sample["relevant"],3)})
        logger.debug("query={} | top={} | mrr={}",sample["query"],ranked[:3],rows[-1]["mrr"])
    from utils.metrics import mean
    save_json_state({"documents": docs, "k1": engine.k1, "b": engine.b})

    return {"task":"retrieval","documents":len(docs),"queries":len(valid),
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
