# 09 · BERT 双向表示与 Masked LM

**时代：** 2018–2019

**问题：** 为什么理解任务需要同时利用 token 左右两侧的上下文？

## 最小模型的核心逻辑

对随机隐藏的 token 做分类；Encoder 无因果掩码，隐藏位置可以看见左右未遮盖 token。

```text
L_MLM = −Σ_(i∈masked) log P(x_i | x_masked)
```

## 逐行阅读指南

1. **构造 Mask**：选位置替换为 [MASK]，仅这些位置参与 loss；在代码中搜索 `inputs[mask_positions] = MASK`。
2. **双向 Encoder**：self-attention 允许读取左右上下文；在代码中搜索 `nn.TransformerEncoder`。
3. **预测原词**：为 masked token 位置输出词表 logits；在代码中搜索 `head(hidden)`。
4. **稀疏损失**：计算原始目标词的 cross entropy；在代码中搜索 `logits[mask_positions]`。

## 运行

```bash
pip install -r requirements.txt
python 09-bert-mlm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/stanfordnlp/sst2
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1810.04805

## 重要边界

本章只教学 MLM，不包含原始 BERT 的 NSP、WordPiece 词表、80/10/10 mask 策略及大型预训练。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 09-bert-mlm/demo.py --steps 30 --output-dir outputs/09-bert-mlm
cat outputs/09-bert-mlm/metrics.json
```

**本章实际可验收任务：** 双向 Masked LM；训练只对 mask 位置计算 CE，验证单独逐位置 mask 并计算准确率。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。
