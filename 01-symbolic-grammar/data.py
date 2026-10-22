"""CFG 的正例和负例分开保存：不允许用 parser 自己生成的答案当测试标签。"""
TRAIN = [
    ("the cat sees a dog", True),
    ("a dog likes the cat", True),
    ("the dog sees the cat", True),
    ("cat sees dog", False),
]
VALID = [
    ("a cat likes a dog", True),
    ("the dog likes a cat", True),
    ("the sees cat", False),
    ("a dog the cat sees", False),
]
