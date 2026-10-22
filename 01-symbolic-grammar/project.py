"""可运行 CFG 任务：加载人工规则，评估接受率，返回结构化解析树。"""
from loguru import logger
from utils.v2io import save_json_state, load_json_state
from data import TRAIN, VALID
from model import GRAMMAR, parse

def run(*, steps=1, seed=42, text_file=None):
    train,valid=TRAIN,VALID
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        data=read_jsonl(text_file)
        if len(data)<4 or not all(isinstance(r.get("sentence"),str) and
             isinstance(r.get("accepted"),bool) for r in data):
            raise ValueError("CFG JSONL 至少 4 条，字段为 sentence:string, accepted:boolean")
        left,right=nonoverlap_split(data)
        train=[(r["sentence"],r["accepted"]) for r in left]
        valid=[(r["sentence"],r["accepted"]) for r in right]
    logger.info("grammar rules={} | train={} | valid={}",len(GRAMMAR),len(train),len(valid))
    # 符号文法不进行梯度学习：fit 即指定产生式/词典的显式知识。
    grammar_vocabulary = sorted({token for productions in GRAMMAR.values()
                                 for production in productions for token in production
                                 if token not in GRAMMAR})
    examples = []
    for sentence, truth in valid:
        trees = parse(sentence)
        pred = bool(trees)
        logger.debug("{} | gold={} predicted={} parses={}",sentence,truth,pred,len(trees))
        examples.append({"text":sentence,"gold":truth,"pred":pred,"parse_count":len(trees),
                         "tree":trees[0] if trees else None})
    accuracy = sum(r["gold"] == r["pred"] for r in examples) / len(examples)
    metrics = {"task":"grammar_acceptability","val_accuracy":accuracy,
               "valid_samples":len(valid),"rules":sum(map(len,GRAMMAR.values())),
               "vocabulary_size":len(grammar_vocabulary)}
    save_json_state({"grammar": GRAMMAR, "vocabulary": grammar_vocabulary})

    return metrics, examples


def predict_saved(directory, text=None):
    """独立进程重新加载文法规则后解析输入，不依赖训练阶段的内存。"""
    import model as grammar_module
    state = load_json_state(directory)
    grammar_module.GRAMMAR = {
        symbol: [tuple(rule) for rule in rules]
        for symbol, rules in state["grammar"].items()
    }
    sentence = text if text is not None else "the cat sees a dog"
    parses = grammar_module.parse(sentence)
    return {"input": sentence, "accepted": bool(parses),
            "parse_count": len(parses), "tree": parses[0] if parses else None}
