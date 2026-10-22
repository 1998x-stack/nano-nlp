"""已标注的极简 POS 数据 (N=名词,V=动词)。注意 held-out 不进入参数估计。"""
TRAIN = [
    (["dogs", "run"], ["N", "V"]),
    (["dogs", "run"], ["N", "V"]),
    (["run", "dogs"], ["V", "N"]),
    (["dogs", "dogs"], ["N", "N"]),
    (["run", "run"], ["V", "V"]),
    (["run", "dogs"], ["V", "N"]),
]
VALID = [
    (["dogs", "run"], ["N", "V"]),
    (["run", "dogs"], ["V", "N"]),
    (["dogs", "dogs"], ["N", "N"]),
]
