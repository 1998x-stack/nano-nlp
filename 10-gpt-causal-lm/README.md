# 10 · nanoGPT 风格的 Decoder-only GPT

**时代：** 2018–2020

**问题：** 为什么 next-token prediction 配合因果注意力，可以训练生成式基础模型？

## 最小模型的核心逻辑

Token + Position Embedding → Causal Attention Block → LM Head；用右移标签训练，生成时只依赖已见 token。

```text
L = −Σ_t log P(x_(t+1) | x_(≤t))
```

## 逐行阅读指南

1. **字符编码**：字符→ID，训练样本 x 与右移目标 y；在代码中搜索 `ids[i:i+T], ids[i+1:i+T+1]`。
2. **位置与词嵌入**：让模型区分同词在不同位置；在代码中搜索 `token(x) + position(arange(T))`。
3. **因果注意力**：下三角可见，防止看到未来标签；在代码中搜索 `mask = triu(1)`。
4. **残差与输出**：Attention + MLP 残差，线性映射 vocab；在代码中搜索 `head(x)`。
5. **优化与生成**：交叉熵训练后，自回归采样下一个 token；在代码中搜索 `F.cross_entropy(logits, y)`。

## 运行

```bash
pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

## 重要边界

本章只实现单层单批极简 GPT，不包含工业级 DDP、FlashAttention、KV cache、BPE 或大规模训练。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 30 --output-dir outputs/10-gpt-causal-lm
cat outputs/10-gpt-causal-lm/metrics.json
```

**本章实际可验收任务：** 字符 GPT：训练集/验证集分段、causal mask、CE、PPL、无监督自回归生成。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。
