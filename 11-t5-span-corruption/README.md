# 11 · T5 Text-to-Text 与 Span Corruption

2020 · **要解决的问题**：如何把预训练目标与翻译、摘要、QA 统一为 text-to-text？

## 一句话理解

Encoder 读入含 sentinel 的损坏文本，Decoder 自回归生成被移除的片段及相应 sentinel。

```text
input: the <extra_id_0> sat ... <extra_id_1> → target: <extra_id_0> cat <extra_id_1> mat <eos>
```

## 代码阅读路线

1. **随机片段移除**：用 sentinel 替换连续片段，而非每 token 随机遮挡。代码锚点：`SOURCE`。
2. **构造目标序列**：目标按 sentinel 顺序重建原始片段。代码锚点：`TARGET`。
3. **教师强制输入**：Decoder 输入是目标右移一位。代码锚点：`DECODER_INPUT`。
4. **交叉注意力**：Decoder 对 Encoder memory 做 cross-attention。代码锚点：`nn.Transformer`。
5. **序列损失**：对所有目标位置计算 token 预测损失。代码锚点：`F.cross_entropy`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 11-t5-span-corruption/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/Salesforce/wikitext

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://www.jmlr.org/papers/volume21/20-074/20-074.pdf

## 局限与误区

原版 T5 包含 SentencePiece、相对位置偏置和精心设计的语料混合；本章只是捕捉 span corruption 和 Encoder–Decoder 目标。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。
