"""从短语料构建语言模型；验证句与训练句不完全相同；按 held-out 下一个词的对数似然评估。"""
TRAIN = [
    "the cat likes fish",
    "the dog likes fish",
    "the cat likes milk",
    "the dog eats fish",
    "the dog likes milk",
    "the cat eats fish",
]
VALID = [
    "the cat eats milk",
    "the dog eats milk",
]
