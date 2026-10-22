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

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 13-rag-retrieval/demo.py --steps 1 --output-dir outputs/13-rag-retrieval
cat outputs/13-rag-retrieval/metrics.json
```

**本章 MVP 的验收任务：** 索引文档并进行检索，输出带来源 ID 的抽取式证据回答；报告查询级证据召回和无证据拒答。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 通过 BM25 构建可重建的本地检索器，对每个查询召回 top-k 文档并给出带 ID 的可核对抽取式回答；无证据时拒答。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → 项目推断逻辑按需从保存的 JSON 恢复统计状态 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 13-rag-retrieval/demo.py --steps 30 --output-dir outputs/13-rag-retrieval
python 13-rag-retrieval/demo.py --mode predict --output-dir outputs/13-rag-retrieval --input 'What does Transformer use?'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** evidence_recall_at_2_with_abstention、检索分数、provenance 与引用一致性。

**最值得做的消融实验：** 把查询替换成语料没有包含的事实，确认 evidence不足而非编造。

**不要过度解读：** 本章不接外部生成 LLM；属于 extractive answer MVP，不冒称完整生成式 RAG。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
