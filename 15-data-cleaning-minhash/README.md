# 15 · LLM 数据清洗、去重与 MinHash

2019–2026 · **要解决的问题**：为什么数据过滤和近似去重会直接影响 NLP 训练质量？

## 一句话理解

先内容标准化、Exact Dedup，再通过词 n-gram 集合 MinHash/LSH 查找高度相似的文档，最后保留代表文本。

```text
P[MinHash(A)=MinHash(B)] = Jaccard(A,B)
```

## 代码阅读路线

1. **规范化**：小写并统一空白，减少格式差异。代码锚点：`normalize(text)`。
2. **精确去重**：以规范化内容哈希过滤完全重复项。代码锚点：`hashlib.sha256(...)`。
3. **局部 shingles**：按词构造集合以比较重叠率。代码锚点：`shingles(text, n=2)`。
4. **MinHash 签名**：多个稳定哈希排列得到集合摘要。代码锚点：`min(hash(seed, shingle))`。
5. **LSH 分桶**：匹配任何一个 band 的签名获得候选。代码锚点：`bands(signature)`。
6. **代表选择**：保留最早的未命中样本；生产需优先级策略。代码锚点：`kept.append(doc)`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 15-data-cleaning-minhash/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/Salesforce/wikitext

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2107.06499

## 局限与误区

MinHash/LSH 是概率近似；本例短文本仅用于机制验证，生产需外部排序、分片和质量分层。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 15-data-cleaning-minhash/demo.py --steps 1 --output-dir outputs/15-data-cleaning-minhash
cat outputs/15-data-cleaning-minhash/metrics.json
```

**本章 MVP 的验收任务：** 质量过滤、精确去重、LSH 近似候选和 Jaccard 复核；输出各文档删除原因与保留代表。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 按质量排序文档、规范化并 hash 查 exact duplicate，再利用 MinHash LSH 候选及 Jaccard 复核决定保留代表文档。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → 项目推断逻辑按需从保存的 JSON 恢复统计状态 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 15-data-cleaning-minhash/demo.py --steps 30 --output-dir outputs/15-data-cleaning-minhash
python 15-data-cleaning-minhash/demo.py --mode predict --output-dir outputs/15-data-cleaning-minhash --input 'NLP models learn from a large text corpus.'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** raw_docs/kept_docs、保留率、删除原因、representative 与 nearest Jaccard。

**最值得做的消融实验：** 改变 shingles 的 n、Jaccard 阈值，记录误删与漏删案例。

**不要过度解读：** LSH 是概率候选机制，模型不能保证召回所有真正近重复文档。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
