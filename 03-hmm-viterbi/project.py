"""由标注样本拟合 HMM；用动态规划解码验证集而非写死矩阵。"""
from collections import Counter, defaultdict
from math import log
from loguru import logger
from utils.v2io import save_json_state, load_json_state
from data import TRAIN, VALID

class FittedHMM:
    def __init__(self, smoothing=0.5):
        self.smoothing=smoothing

    def fit(self, samples):
        self.states=sorted({tag for words,tags in samples for tag in tags})
        self.vocab=sorted({w for words,tags in samples for w in words})
        self.initial=Counter()
        self.transition=defaultdict(Counter)
        self.emission=defaultdict(Counter)
        for words,tags in samples:
            if not words or len(words)!=len(tags):
                raise ValueError("tokens/tags 必须非空且长度一致")
            self.initial[tags[0]] += 1
            for i, (word,tag) in enumerate(zip(words,tags)):
                self.emission[tag][word]+=1
                if i>0:
                    self.transition[tags[i-1]][tag]+=1
        return self

    def _log(self,counter,key,universe):
        alpha=self.smoothing
        return log((counter[key]+alpha)/(sum(counter.values())+alpha*universe))

    def predict(self,words):
        if not words:
            return [], []
        dp={s:self._log(self.initial,s,len(self.states))+
              self._log(self.emission[s],words[0],len(self.vocab)+1) for s in self.states}
        grid=[dict(dp)]
        prevs=[]
        for word in words[1:]:
            next_dp,back={},{}
            for state in self.states:
                previous=max(self.states,key=lambda p:dp[p]+self._log(self.transition[p],state,len(self.states)))
                next_dp[state]=dp[previous]+self._log(self.transition[previous],state,len(self.states)) + self._log(self.emission[state],word,len(self.vocab)+1)
                back[state]=previous
            dp=next_dp
            grid.append(dict(dp))
            prevs.append(back)
        best=max(self.states,key=dp.get)
        path=[best]
        for back in reversed(prevs):
            path.append(back[path[-1]])
        return list(reversed(path)),grid

def run(*, steps=1, seed=42, text_file=None):
    hmm=FittedHMM().fit(TRAIN)
    logger.info("fit HMM | states={} vocab={} labeled_sequences={}",hmm.states,len(hmm.vocab),len(TRAIN))
    rows=[]
    total=correct=0
    for words,tags in VALID:
        pred,grid=hmm.predict(words)
        total+=len(tags);correct+=sum(a==b for a,b in zip(tags,pred))
        rows.append({"tokens":words,"gold":tags,"pred":pred,
                     "viterbi_scores":[{k:round(v,5) for k,v in row.items()} for row in grid]})
    save_json_state({"states": hmm.states, "vocab": hmm.vocab, "smoothing": hmm.smoothing,
                     "initial": dict(hmm.initial),
                     "transition": {k: dict(v) for k,v in hmm.transition.items()},
                     "emission": {k: dict(v) for k,v in hmm.emission.items()}})

    return {"task":"pos_tagging","states":hmm.states,"train_sequences":len(TRAIN),
            "val_token_accuracy":correct/total,"val_tokens":total},rows


def predict_saved(directory, text=None):
    """从计数表重构同一个 HMM，再进行 Viterbi 解码。"""
    from collections import Counter, defaultdict
    data = load_json_state(directory)
    hmm = FittedHMM(smoothing=data["smoothing"])
    hmm.states, hmm.vocab = data["states"], data["vocab"]
    hmm.initial = Counter(data["initial"])
    hmm.transition = defaultdict(Counter, {k: Counter(v) for k,v in data["transition"].items()})
    hmm.emission = defaultdict(Counter, {k: Counter(v) for k,v in data["emission"].items()})
    tokens = (text or "dogs run").lower().split()
    tags, scores = hmm.predict(tokens)
    return {"tokens": tokens, "tags": tags, "last_log_scores": scores[-1] if scores else {}}
