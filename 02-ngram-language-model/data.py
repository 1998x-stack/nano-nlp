"""从短语料构建语言模型；held-out 集仅含训练词表中的词。"""
TRAIN = [
    "the cat likes fish",
    "the dog likes fish",
    "the cat likes milk",
    "the dog eats fish",
    "the dog likes milk",
    "the cat eats fish",
]
VALID = [
    "the cat likes fish",
    "the dog eats fish",
]
