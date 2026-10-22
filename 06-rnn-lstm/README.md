# 06 · RNN 与 LSTM 序列语言模型

**时代：** 1997 / 2014

**问题：** 如何让当前词的预测依赖此前的信息，而不再局限于 N-gram？

## 最小模型的核心逻辑

Embedding → LSTM(hidden,cell) → Linear → Next-token Loss；门结构调节记忆，仍然逐时间步递归。

```text
(h_t,c_t) = LSTM(x_t, h_(t−1), c_(t−1))
```

## 逐行阅读指南

1. **词表与序列**：将每个字符映射到整数，构造输入 x 与右移标签 y；在代码中搜索 `ids[:-1], ids[1:]`。
2. **嵌入**：索引向量比 one-hot 更紧凑且可训练；在代码中搜索 `nn.Embedding(vocab, 24)`。
3. **门控记忆**：LSTM 维护隐藏状态 h 与记忆状态 c；在代码中搜索 `self.rnn(self.embed(x))`。
4. **预测训练**：使用每个时间步的 logits 计算交叉熵；在代码中搜索 `F.cross_entropy(logits, target)`。

## 运行

```bash
pip install -r requirements.txt
python 06-rnn-lstm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://www.bioinf.jku.at/publications/older/2604.pdf

## 重要边界

LSTM 改善梯度流动却没有消除沿时间的串行计算。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。
