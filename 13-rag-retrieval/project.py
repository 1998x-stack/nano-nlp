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
    docs,valid=DOCS,VALID
    if text_file:
        from utils.data import typed_records
        source=typed_records(text_file,{"document","query"})
        docs=[{"id":r["id"],"text":r["text"]} for r in source if r["type"]=="document"]
        valid=[{"question":r["question"],"evidence":r["evidence"]}
               for r in source if r["type"]=="query"]
        ids={d["id"] for d in docs}
        if not docs or not valid or len(ids)!=len(docs):
            raise ValueError("RAG 输入需要唯一 id 的 document 与 query")
        if not all(isinstance(r["text"],str) and r["text"].strip() for r in docs):
            raise ValueError("document.text 不能为空")
        if not all(isinstance(r["question"],str) and isinstance(r["evidence"],list)
                   and set(r["evidence"]) <= ids for r in valid):
            raise ValueError("evidence 必须引用已有 document.id")
    engine=BM25([d["text"] for d in docs])
    rows=[]
    for item in valid:
        result=answer(item["question"],engine,docs,k=2)
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
    save_json_state({"documents": docs, "generator": "extractive_evidence_only", "k": 2})

    return {"task":"extractive_rag","documents":len(docs),"queries":len(valid),
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
