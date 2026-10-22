"""极小 CFG 递归下降解析：显式展示规则，不依赖 NLP 框架。"""
GRAMMAR = {
    "S": [("NP", "VP")],
    "NP": [("Det", "N")],
    "VP": [("V", "NP")],
    "Det": [("the",), ("a",)],
    "N": [("cat",), ("dog",)],
    "V": [("sees",), ("likes",)],
}

def parse_symbol(symbol, tokens, pos):
    """返回所有 (树, 下一个索引)，支持多个产生式。"""
    if symbol not in GRAMMAR:  # 终结符：直接与输入匹配
        return [(symbol, pos + 1)] if pos < len(tokens) and tokens[pos] == symbol else []
    results = []
    for production in GRAMMAR[symbol]:
        states = [([], pos)]
        for child in production:
            next_states = []
            for children, offset in states:
                for tree, new_offset in parse_symbol(child, tokens, offset):
                    next_states.append((children + [tree], new_offset))
            states = next_states
        for children, offset in states:
            results.append(((symbol, children), offset))
    return results

def parse(sentence):
    tokens = sentence.lower().split()
    return [tree for tree, end in parse_symbol("S", tokens, 0) if end == len(tokens)]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", default="the cat sees a dog")
    args = p.parse_args()
    trees = parse(args.sentence)
    print("RESULT:", trees[0] if trees else "无法被当前文法解析")
