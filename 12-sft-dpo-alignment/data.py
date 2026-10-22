"""两种固定 prompt 的玩具偏好分类，不是 held-out prompt 泛化测试。

训练和验证复用 prompt-id（不是独立问题特征），指标需标记 seen_prompts。
"""
TRAIN=[(0,1,0),(1,0,1)]
VALID=[(0,1,0),(1,0,1)]
