"""教学版 Exact + MinHash + LSH。不会调用外部大模型。"""
import hashlib
import re

DOCS = [
    "NLP models learn from a large text corpus.",
    "NLP models learn from a large text corpus.",
    "NLP models learn from large text corpora.",
    "Attention connects different tokens in a sequence.",
]

def normalize(text):
    return " ".join(re.findall(r"[a-z]+", text.lower()))

def shingles(text, n=2):
    words = normalize(text).split()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

def stable_hash(seed, token):
    payload = f"{seed}:{token}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

def minhash(text, permutations=24):
    tokens = shingles(text) or {normalize(text)}
    return tuple(min(stable_hash(i, token) for token in tokens)
                 for i in range(permutations))

def bands(signature, rows=3):
    assert len(signature) % rows == 0
    return [signature[i:i+rows] for i in range(0, len(signature), rows)]

def deduplicate(documents):
    seen_exact, seen_bands, retained = set(), set(), []
    for text in documents:
        exact = hashlib.sha256(normalize(text).encode()).hexdigest()
        if exact in seen_exact:
            continue
        seen_exact.add(exact)
        signature = minhash(text)
        keys = {(i, band) for i, band in enumerate(bands(signature))}
        if seen_bands.intersection(keys):  # LSH 高相似候选
            continue
        retained.append(text)
        seen_bands.update(keys)
    return retained

if __name__ == "__main__":
    print("RESULT:", deduplicate(DOCS))
