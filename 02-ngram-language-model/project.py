"""可运行 Bigram 任务：估计条件概率、验证 NLL 与生成。"""
import math
from loguru import logger
from data import TRAIN, VALID
from model import BigramLM

def generate(lm, start="the", max_tokens=8):
    """贪心生成：每一步只用之前一个 token。"""
    result = [start]
    for _ in range(max_tokens):
        candidates = lm.predict(result[-1])
        word = candidates[0][0]
        if word == "<eos>":
            break
        result.append(word)
    return " ".join(result)

def run(*, steps=1, seed=42, text_file=None):
    train = list(TRAIN)
    valid = list(VALID)
    if text_file:
        from utils.data import read_text
        sents = [line.strip().lower() for line in read_text(text_file).splitlines() if len(line.split()) >= 3]
        if len(sents) < 5:
            raise ValueError("外部文件需至少 5 行，每行 3 个词")
        train, valid = sents[:-2], sents[-2:]
    logger.info("fitting bigram counts: train={} valid={}",len(train),len(valid))
    model = BigramLM().fit(train)
    examples = []
    total_nll, tokens = 0., 0
    for sentence in valid:
        prev = "<bos>"
        nll = 0.0
        for word in sentence.lower().split() + ["<eos>"]:
            # 对 OOV 明确提示，不把未见词的概率误当已训练词表归一化概率。
            if word not in model.vocab:
                logger.warning("OOV token '{}' in validation; smoothed score is heuristic",word)
            prob = model.probability(prev,word)
            nll -= math.log(prob)
            tokens += 1
            prev = word
        total_nll += nll
        examples.append({"sentence":sentence,"nll":round(nll,5),"perplexity":round(math.exp(nll/(len(sentence.split())+1)),5)})
    avg_nll = total_nll / tokens
    generated = generate(model)
    logger.info("vocab={} avg_nll={:.4f} generated={!r}",len(model.vocab),avg_nll,generated)
    return {"task":"next_word_prediction","train_samples":len(train),"valid_samples":len(valid),
            "vocab_size":len(model.vocab),"val_nll":round(avg_nll,6),
            "val_perplexity":round(math.exp(avg_nll),5),"generated":generated}, examples
