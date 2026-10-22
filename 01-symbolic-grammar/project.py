"""可运行 CFG 任务：加载人工规则，评估接受率，返回结构化解析树。"""
from loguru import logger
from data import TRAIN, VALID
from model import GRAMMAR, parse

def run(*, steps=1, seed=42, text_file=None):
    logger.info("grammar rules={} | train={} | valid={}",len(GRAMMAR),len(TRAIN),len(VALID))
    # 符号文法不进行梯度学习：fit 即指定产生式/词典的显式知识。
    grammar_vocabulary = sorted({token for productions in GRAMMAR.values()
                                 for production in productions for token in production
                                 if token not in GRAMMAR})
    examples = []
    for sentence, truth in VALID:
        trees = parse(sentence)
        pred = bool(trees)
        logger.debug("{} | gold={} predicted={} parses={}",sentence,truth,pred,len(trees))
        examples.append({"text":sentence,"gold":truth,"pred":pred,"parse_count":len(trees),
                         "tree":trees[0] if trees else None})
    accuracy = sum(r["gold"] == r["pred"] for r in examples) / len(examples)
    metrics = {"task":"grammar_acceptability","val_accuracy":accuracy,
               "valid_samples":len(VALID),"rules":sum(map(len,GRAMMAR.values())),
               "vocabulary_size":len(grammar_vocabulary)}
    return metrics, examples
