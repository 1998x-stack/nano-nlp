# 08 · Transformer Self-Attention 机制

**时代：** 2017

**问题：** 如何去掉 RNN，使任意两个 token 可以直接交互并进行并行计算？

## 最小模型的核心逻辑

Q/K/V 线性投影、缩放点积、Softmax、因果掩码、多头拼接，是 Decoder Transformer 的计算内核。

```text
softmax(Q Kᵀ / √d_k + causal_mask) V
```

## 逐行阅读指南

1. **生成 QKV**：共享输入通过三组线性投影形成 query/key/value；在代码中搜索 `self.qkv(x).chunk(3, dim=-1)`。
2. **拆分多头**：把通道拆为 heads × head_dim；在代码中搜索 `reshape(B, T, H, D).transpose(1,2)`。
3. **注意力分数**：点积并除 sqrt(head_dim) 防止规模过大；在代码中搜索 `(q @ k.transpose(-2,-1))/sqrt(D)`。
4. **因果掩码**：禁止当前位置看未来 token；在代码中搜索 `masked_fill(mask, -inf)`。
5. **加权汇总**：softmax 后乘以 V，并拼接各头；在代码中搜索 `weights @ value`。

## 运行

```bash
pip install -r requirements.txt
python 08-transformer-attention/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1706.03762

## 重要边界

Self-Attention 本身不包含位置概念；需要显式位置编码或其他位置机制。本章专注 MHA 算子。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。
