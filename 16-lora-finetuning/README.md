# 16 · LoRA 参数高效微调

2021–2023 · **要解决的问题**：微调 LLM 时，为什么可以只训练极少量的低秩参数？

## 一句话理解

冻结预训练权重 W，只学习低秩增量 BA 并通过缩放系数叠加到线性层输出。

```text
y = Wx + (α/r)BAx
```

## 代码阅读路线

1. **冻结基座**：W 的 requires_grad=False，避免全参数更新。代码锚点：`self.base.weight.requires_grad_(False)`。
2. **低秩矩阵**：训练 A: in→rank，B: rank→out。代码锚点：`self.A, self.B`。
3. **前向叠加**：base(x) + B(A(x))*alpha/r。代码锚点：`self.base(x) + self.B(self.A(x))*...`。
4. **只优化 adapter**：优化器仅接收 A、B 参数。代码锚点：`optimizer = AdamW([*lora.A.parameters(), ...])`。
5. **统计可训练比**：对比可训练参数与总参数。代码锚点：`trainable, total`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 16-lora-finetuning/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/stanfordnlp/sst2

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2106.09685

## 局限与误区

真正的 LoRA 微调需要选择注意力/FFN 投影、考虑 rank 与任务、保存 adapter，并与量化和 tokenizer 配合。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。
