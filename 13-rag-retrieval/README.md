# 13 · RAG 检索增强生成骨架

2020–2026 · **要解决的问题**：如何让模型使用外部可追踪事实，而不是只依赖参数记忆？

## 一句话理解

Query → BM25 Retrieve → Top-k Evidence → Prompt Construction → (optional) Generator；先保证证据与引用可检查。

```text
P(y|q) ≈ Σ_d P(d|q) · P(y|q,d)
```

## 代码阅读路线

1. **索引语料**：本例使用带 document id 的本地短文。代码锚点：`CORPUS`。
2. **召回证据**：BM25 将问题和文档映射到分数。代码锚点：`BM25(...).search(query)`。
3. **截取 top-k**：限制证据数量以控制上下文长度。代码锚点：`hits[:k]`。
4. **保留出处**：把 [D1] 等来源插入生成提示词。代码锚点：`f'[{doc_id}]...'`。
5. **输出约束**：提示模型只基于证据作答，并给出引用。代码锚点：`build_prompt(query, hits)`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 13-rag-retrieval/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/rajpurkar/squad

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2005.11401

## 局限与误区

本章没有接入外部生成式模型，因此输出是带引用的证据提示词，不是假装已经生成了高质量答案。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。
