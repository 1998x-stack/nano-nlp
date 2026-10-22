# 05 · Word2Vec Skip-gram 与负采样

> 2013 · 怎样从无监督上下文共现中学出分布式词向量？

**一句话抓住核心：** 给定中心词，最大化真实上下文向量内积的 sigmoid，并降低负采样词向量内积。

## 运行 (先离线后扩展)

```bash
python -m pip install -r requirements.txt
python 05-word2vec-skipgram/demo.py --steps 30
python -m http.server 8000  # 访问 /05-word2vec-skipgram/index.html
```

## 从输入到输出的关键步骤

1. **生成词对**：滑动窗口创建 (center, context) 正样本。代码锚点：`make_pairs(tokens, window=2)`
2. **词向量表**：中心词与上下文词各有一个可学习矩阵。代码锚点：`nn.Embedding(vocab, dim)`
3. **负采样**：随机抽取非真实上下文词，构造对比信号。代码锚点：`torch.randint(...)`
4. **优化目标**：增大正样本相似度、压低负样本相似度。代码锚点：`softplus(-positive)+softplus(negative)`

## 核心公式

```text
L=−log σ(v_c·u_o)−Σ_k log σ(−v_c·u_n)
```

## 为什么它比前一代有效？

给定中心词，最大化真实上下文向量内积的 sigmoid，并降低负采样词向量内积。

## 重点阅读代码

推荐先读 [data.py](./data.py) 的任务样本、[model.py](./model.py) 的核心公式实现，再看 [project.py](./project.py) 的训练/验证/保存流程。最后运行 [demo.py](./demo.py)：它只是 loguru CLI 入口，不是算法本体。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/Salesforce/wikitext
- 使用建议：WikiText-2，取前几千文本行训练小词表；演示默认只用几句 toy 语料。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://arxiv.org/pdf/1310.4546 (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** 负样本概率分布会影响学习质量；本章教学用均匀采样，论文常用 unigram^0.75。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。

## v2.0 · 完整 MVP 入口

本章现已拆分为 `data.py`（训练/验证样本与边界）、`model.py`（核心算法）、`project.py`（任务训练/验证/推理）、`demo.py`（运行命令与 loguru）。输出包括 `metrics.json`、`examples.jsonl` 和 `run.log`，且**不在模块导入时执行训练**。

```bash
pip install -r requirements.txt
python 05-word2vec-skipgram/demo.py --steps 30 --output-dir outputs/05-word2vec-skipgram
cat outputs/05-word2vec-skipgram/metrics.json
```

**独立验证的定义**：使用与训练词对不同的验证词对计算 negative-sampling loss，并提取词向量 cosine 近邻。

**实验建议**：改变输入集、训练步数（统计方法修改数据而非增加无意义 epochs）、种子，并比较工件中每条结果，而不仅看一个标量。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 训练两张 token embedding；最大化正上下文的 σ(v·u) 并使负样本内积减小。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 05-word2vec-skipgram/demo.py --steps 30 --output-dir outputs/05-word2vec-skipgram
python 05-word2vec-skipgram/demo.py --mode predict --output-dir outputs/05-word2vec-skipgram --input 'cat'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_negative_sampling_loss、cosine 近邻、不包含自身的 top-k 列表。

**关键正确性断言：** 保存词表与 center/context 两张表后，在独立进程查询同一词。

**建议的对照实验：** 扩大上下文窗口，对正例数量和近邻关系做消融；比较随机种子。

**能力边界：** 负采样仅为局部分类目标，本课程均匀采负样本，不等价论文中的 unigram^0.75。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
