# 07 · Seq2Seq 与 Bahdanau 风格 Attention

**时代：** 2014–2015

**问题：** 为什么翻译长句时，Decoder 不该只依赖单个固定长度向量？

## 最小模型的核心逻辑

Encoder 提供每个位置的隐藏状态，Decoder 在每一步动态计算注意力权重，并加权组合输入上下文。

```text
α_t=softmax(score(h_dec,h_enc_i)); c_t=Σ α_(t,i) h_enc_i
```

## 逐行阅读指南

1. **编码源句**：GRU 输出全部 encoder hidden states；在代码中搜索 `memory, hidden = encoder(source)`。
2. **计算对齐**：Decoder hidden 与源端各位置相似度；在代码中搜索 `scores = memory @ hidden`。
3. **动态上下文**：softmax 得到权重并求加权和；在代码中搜索 `context = attention @ memory`。
4. **自回归解码**：预测目标词，teacher forcing 提供下一个输入；在代码中搜索 `target[:, t]`。

## 运行

```bash
pip install -r requirements.txt
python 07-seq2seq-attention/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/Helsinki-NLP/opus_books
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1409.0473

## 重要边界

这是教学级 dot-product cross-attention，而非 Bahdanau 论文中的 additive score；两者都缓解固定向量瓶颈。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。
