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
