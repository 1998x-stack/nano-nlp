# 04 · TF-IDF 与 BM25 信息检索

> 1970s–2000s · 为什么一个词既要看文档内频率，也要看整个语料中的稀缺程度？

**一句话抓住核心：** BM25 综合词频饱和、文档长度归一化和逆文档频率，代表了可解释的传统搜索排序方法。

## 运行 (先离线后扩展)

```bash
python -m pip install -r requirements-classic.txt
python 04-tfidf-bm25-search/demo.py 
python -m http.server 8000  # 访问 /04-tfidf-bm25-search/index.html
```

## 从输入到输出的关键步骤

1. **建索引**：词项映射到包含它的文档集合。代码锚点：`df[token]`
2. **计算 IDF**：越少文档包含该词，区分能力越强。代码锚点：`log(1+(N-df+0.5)/(df+0.5))`
3. **词频饱和**：同一词重复出现的边际收益递减。代码锚点：`tf*(k1+1)/(tf+K)`
4. **长度校正**：长文不应仅因更长而获得更高分。代码锚点：`K=k1*(1-b+b*dl/avgdl)`

## 核心公式

```text
score=Σ IDF(q)·tf(k1+1)/(tf+k1(1-b+b·dl/avgdl))
```

## 为什么它比前一代有效？

BM25 综合词频饱和、文档长度归一化和逆文档频率，代表了可解释的传统搜索排序方法。

## 重点阅读代码

推荐先读 [data.py](./data.py) 的任务样本、[model.py](./model.py) 的核心公式实现，再看 [project.py](./project.py) 的训练/验证/保存流程。最后运行 [demo.py](./demo.py)：它只是 loguru CLI 入口，不是算法本体。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/rajpurkar/squad
- 使用建议：SQuAD 的 context 可作为小型可检索文档库、question 作为查询；先运行内置的3篇短文。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** BM25 不理解同义改写或语义；但作为 Hybrid Search 的稀疏分支依然非常实用。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。

## v2.0 · 完整 MVP 入口

本章现已拆分为 `data.py`（训练/验证样本与边界）、`model.py`（核心算法）、`project.py`（任务训练/验证/推理）、`demo.py`（运行命令与 loguru）。输出包括 `metrics.json`、`examples.jsonl` 和 `run.log`，且**不在模块导入时执行训练**。

```bash
pip install -r requirements.txt
python 04-tfidf-bm25-search/demo.py --steps 1 --output-dir outputs/04-tfidf-bm25-search
cat outputs/04-tfidf-bm25-search/metrics.json
```

**独立验证的定义**：用固定查询的人工相关性标签计算 MRR 与 Recall@3，另输出分数最高的文档。

**实验建议**：改变输入集、训练步数（统计方法修改数据而非增加无意义 epochs）、种子，并比较工件中每条结果，而不仅看一个标量。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 构建文档词频和文档频率，按查询计算 IDF、词频饱和及长度校正；对返回的 top-k 保留原始文档 ID。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 04-tfidf-bm25-search/demo.py --steps 30 --output-dir outputs/04-tfidf-bm25-search
python 04-tfidf-bm25-search/demo.py --mode predict --output-dir outputs/04-tfidf-bm25-search --input 'attention language'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** MRR、Recall@3、query 对每篇文档的 BM25 score。

**关键正确性断言：** 从 checkpoint.json 恢复文档并重建索引后应获得同样排序。

**建议的对照实验：** 改变 b∈[0,1]、k1 的大小观察长文和高词频文档的相对次序。

**能力边界：** BM25 无法依赖语义推断同义词，而且极短语料的 IDF 只是机制演示。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
