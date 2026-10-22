# 05 · Word2Vec Skip-gram 与负采样

> 2013 · 怎样从无监督上下文共现中学出分布式词向量？

**一句话抓住核心：** 给定中心词，最大化真实上下文向量内积的 sigmoid，并降低负采样词向量内积。

## 运行 (先离线后扩展)

```bash
python 05-word2vec-skipgram/demo.py --steps 30
python -m http.server 8000  # 访问 /05-word2vec-skipgram/index.html
```

## 从输入到输出的关键步骤

1. **生成词对**：滑动窗口创建 (center, context) 正样本。代码锚点：`make_pairs(tokens, window=2)`
2. **词向量表**：中心词与上下文词各有一个可学习矩阵。代码锚点：`nn.Embedding(vocab, dim)`
3. **负采样**：随机抽取非真实上下文词，构造对比信号。代码锚点：`torch.randint(...)`
4. **优化目标**：增大正样本相似度、压低负样本相似度。代码锚点：`softplus(-positive)+softplus(negative)`

## 核心公式

```text
L=−log σ(v_c·u_o)−Σ_k log σ(−v_c·u_n)
```

## 为什么它比前一代有效？

给定中心词，最大化真实上下文向量内积的 sigmoid，并降低负采样词向量内积。

## 重点阅读代码

打开 [demo.py](./demo.py)，先读数据定义，再读核心函数，最后读 `if __name__ == "__main__"`。关键算法旁有中文注释；不需要先学复杂工程框架。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/Salesforce/wikitext
- 使用建议：WikiText-2，取前几千文本行训练小词表；演示默认只用几句 toy 语料。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://arxiv.org/pdf/1310.4546 (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** 负样本概率分布会影响学习质量；本章教学用均匀采样，论文常用 unigram^0.75。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。
