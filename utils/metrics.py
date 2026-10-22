"""微型 NLP 任务共享指标，不能用训练损失冒充评估指标。"""
import math

def accuracy(actual, predicted):
    actual, predicted = list(actual), list(predicted)
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual/predicted 应长度相等且非空")
    return sum(a == b for a, b in zip(actual, predicted)) / len(actual)

def perplexity(negative_log_likelihood):
    """输入必须是每 token 的平均自然对数 NLL。"""
    if negative_log_likelihood < 0:
        raise ValueError("NLL 不能为负")
    return math.exp(negative_log_likelihood)

def mean(values):
    x = list(values)
    if not x:
        raise ValueError("empty metric sequence")
    return sum(x) / len(x)

def mrr(ranked_ids, relevant):
    """Mean reciprocal rank of first relevant document; 单个 query 版本。"""
    relevant = set(relevant)
    for i, item in enumerate(ranked_ids, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0

def recall_at_k(ranked_ids, relevant, k):
    if k < 1:
        raise ValueError("k >=1")
    relevant = set(relevant)
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.
