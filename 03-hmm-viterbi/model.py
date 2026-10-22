"""HMM 词性标注：Viterbi 在 log 空间避免概率连乘下溢。"""
from math import log

STATES = ["N", "V"]
START = {"N": 0.8, "V": 0.2}
TRANS = {"N": {"N": 0.2, "V": 0.8}, "V": {"N": 0.8, "V": 0.2}}
EMIT = {
    "N": {"dogs": 0.8, "run": 0.2},
    "V": {"dogs": 0.1, "run": 0.9},
}

def viterbi(words):
    """返回最优标签及其 log 概率。"""
    if not words:
        return [], 0.0
    score = {s: log(START[s]) + log(EMIT[s][words[0]]) for s in STATES}
    backpointers = []
    for word in words[1:]:
        new_score, back = {}, {}
        for cur in STATES:
            # 对每个可能当前标签，只保留最高概率的前驱。
            prev = max(STATES, key=lambda p: score[p] + log(TRANS[p][cur]))
            new_score[cur] = score[prev] + log(TRANS[prev][cur]) + log(EMIT[cur][word])
            back[cur] = prev
        score = new_score
        backpointers.append(back)
    last = max(STATES, key=score.get)
    path = [last]
    for back in reversed(backpointers):
        path.append(back[path[-1]])
    return list(reversed(path)), score[last]

if __name__ == "__main__":
    print("RESULT:", viterbi(["dogs", "run"]))
