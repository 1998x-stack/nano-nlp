"""完整的本地 extractive RAG：索引、召回、证据回答和验证引用。"""
from loguru import logger
from utils.v2io import save_json_state, load_json_state
from utils.retrieval import BM25
from utils.metrics import mean,recall_at_k,mrr
from data import DOCS,VALID

def answer(question,engine,documents,k=2):
    ranked=engine.search(question)
    hits=[(documents[i],engine.score(question,i)) for i in ranked[:k]]
    if not hits or hits[0][1]<=0:
        return {"answer":"证据不足","references":[],"retrieved":[]}
    # 本章选用可检查的抽取式生成：直接返回证据原文，绝不伪造 LLM。
    doc,score=hits[0]
    return {"answer":doc["text"],"references":[doc["id"]],
            "retrieved":[{"id":d["id"],"score":round(float(s),5)} for d,s in hits]}

def run(*,steps=1,seed=42,text_file=None):
    engine=BM25([d["text"] for d in DOCS])
    rows=[]
    for item in VALID:
        result=answer(item["question"],engine,DOCS,k=2)
        ranking=[x["id"] for x in result["retrieved"]]
        relevant=item["evidence"]
        if not relevant:
            correct_abstention=(result["answer"]=="证据不足")
            score=1. if correct_abstention else 0.
        else:
            score=recall_at_k(ranking,relevant,2)
        rows.append({"query":item["question"],"expected_evidence":relevant,
                     **result,"evidence_recall":score})
        logger.debug("query={} | cited={} | recall={}",item["question"],result["references"],score)
    save_json_state({"documents": DOCS, "generator": "extractive_evidence_only", "k": 2})

    return {"task":"extractive_rag","documents":len(DOCS),"queries":len(VALID),
            "evidence_recall_at_2_with_abstention":round(mean(r["evidence_recall"] for r in rows),5),
            "abstentions":sum(r["answer"]=="证据不足" for r in rows)},rows


def predict_saved(directory, text=None):
    """恢复索引后给出有 provenance 的抽取式证据，不冒充生成模型。"""
    state = load_json_state(directory)
    docs = state["documents"]
    query = text or "What does Transformer use?"
    engine = BM25([doc["text"] for doc in docs])
    result = answer(query, engine, docs, k=state["k"])
    return {"query": query, **result, "generator": state["generator"]}
