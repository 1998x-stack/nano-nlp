"""数据清洗的实际 MVP：质量门槛→Exact Dedup→LSH 候选→Jaccard 复核→血缘。"""
import hashlib
from loguru import logger
from data import RAW
from model import normalize,shingles,minhash,bands

def jaccard(left,right):
    a,b=shingles(left),shingles(right)
    return len(a & b)/len(a | b) if a or b else 1.

def run(*,steps=1,seed=42,text_file=None):
    # 高质量代表先进入候选簇，防止随机输入顺序保留较差版本。
    ordered=sorted(RAW,key=lambda d:(-d["quality"],d["id"]))
    signatures={}
    seen_exact={}
    kept=[]
    rows=[]
    for doc in ordered:
        text=doc["text"]
        reason="accepted";rep=None
        if doc["quality"] < 0.3:
            reason="quality_below_threshold"
        else:
            normalized=normalize(text)
            exact=hashlib.sha256(normalized.encode()).hexdigest()
            if exact in seen_exact:
                reason="exact_duplicate";rep=seen_exact[exact]
            else:
                signature=minhash(text)
                keys={(i,band) for i,band in enumerate(bands(signature))}
                # LSH 只找候选：再确认 Jaccard 防误杀。
                for prev in kept:
                    if signatures[prev["id"]] & keys and jaccard(text,prev["text"]) >= .70:
                        reason="near_duplicate"
                        rep=prev["id"]
                        break
                if reason=="accepted":
                    seen_exact[exact]=doc["id"]
                    signatures[doc["id"]]=keys
                    kept.append(doc)
        rows.append({"id":doc["id"],"source":doc["source"],"quality":doc["quality"],
                     "status":reason,"representative":rep})
        logger.debug("doc={} status={} representative={}",doc["id"],reason,rep)
    return {"task":"text_curation","raw_docs":len(RAW),"kept_docs":len(kept),
            "removed_docs":len(RAW)-len(kept),
            "dedup_retention_rate":round(len(kept)/len(RAW),5),
            "retained_ids":[doc["id"] for doc in kept]},rows
