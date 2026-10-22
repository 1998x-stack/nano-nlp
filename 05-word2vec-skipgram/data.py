"""Word2Vec 的极小语料。保留重叠词组，便于词向量近邻解释。"""
TEXT = ("cat likes milk dog likes bones cat likes fish dog likes fish "
        "kitten likes milk puppy likes bones cat eats fish dog eats bones").split()
CONTEXT_WINDOW = 2
