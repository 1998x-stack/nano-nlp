# 04 · TF-IDF 与 BM25 信息检索

> 1970s–2000s · 为什么一个词既要看文档内频率，也要看整个语料中的稀缺程度？

**一句话抓住核心：** BM25 综合词频饱和、文档长度归一化和逆文档频率，代表了可解释的传统搜索排序方法。

## 运行 (先离线后扩展)

```bash
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

打开 [demo.py](./demo.py)，先读数据定义，再读核心函数，最后读 `if __name__ == "__main__"`。关键算法旁有中文注释；不需要先学复杂工程框架。

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
