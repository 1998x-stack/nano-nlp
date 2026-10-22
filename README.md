# nano-nlp — 从规则 NLP 到 LLM 的可运行演进教材

> **学习目标：不仅知道模型名字，还能用最小代码重现它解决的关键问题。**
>
> 灵感来自 [Andrej Karpathy 的 nanoGPT](https://github.com/karpathy/nanoGPT)：优先透明代码、可复现 toy demo、独立数据准备、可阅读的训练循环。这里是教学实现，不是 nanoGPT 的复制品，也不保证生产级性能。

## 30 秒开始

```bash
git clone https://github.com/1998x-stack/nano-nlp.git
cd nano-nlp
git checkout v2.0  # 查看完整 MVP 改造分支
python -m pip install -r requirements-classic.txt  # 经典 NLP 仅需 loguru
python 01-symbolic-grammar/demo.py
python 04-tfidf-bm25-search/demo.py
python -m pip install -r requirements.txt          # 神经章节另需 PyTorch
python 10-gpt-causal-lm/demo.py --steps 25
python -m unittest discover -s tests -v
python -m http.server 8000             # 浏览器访问 /10-gpt-causal-lm/index.html
```

Python 3.10+。Torch 仅对神经网络章节必要；经典算法核心只用 Python 标准库，但完整实验 CLI 需要 loguru。**所有章节默认离线运行，无需 GPU、API Key 或下载大型模型。**

## NLP 历史：17 个可执行实验

| 阶段 | 技术主题 | README | 互动图 | 代码 |
|---|---|---|---|---|
| 01 · 符号主义 | 规则与 CFG | [阅读](01-symbolic-grammar/README.md) | [打开](01-symbolic-grammar/index.html) | [demo](01-symbolic-grammar/demo.py) |
| 02 · 统计 NLP | N-gram 语言模型 | [阅读](02-ngram-language-model/README.md) | [打开](02-ngram-language-model/index.html) | [demo](02-ngram-language-model/demo.py) |
| 03 · 统计 NLP | HMM / Viterbi | [阅读](03-hmm-viterbi/README.md) | [打开](03-hmm-viterbi/index.html) | [demo](03-hmm-viterbi/demo.py) |
| 04 · 搜索与 IR | BM25 检索 | [阅读](04-tfidf-bm25-search/README.md) | [打开](04-tfidf-bm25-search/index.html) | [demo](04-tfidf-bm25-search/demo.py) |
| 05 · 词嵌入 | Word2Vec | [阅读](05-word2vec-skipgram/README.md) | [打开](05-word2vec-skipgram/index.html) | [demo](05-word2vec-skipgram/demo.py) |
| 06 · 序列模型 | LSTM | [阅读](06-rnn-lstm/README.md) | [打开](06-rnn-lstm/index.html) | [demo](06-rnn-lstm/demo.py) |
| 07 · 神经机器翻译 | Seq2Seq Attention | [阅读](07-seq2seq-attention/README.md) | [打开](07-seq2seq-attention/index.html) | [demo](07-seq2seq-attention/demo.py) |
| 08 · Transformer | Transformer Attention | [阅读](08-transformer-attention/README.md) | [打开](08-transformer-attention/index.html) | [demo](08-transformer-attention/demo.py) |
| 09 · 预训练理解 | BERT MLM | [阅读](09-bert-mlm/README.md) | [打开](09-bert-mlm/index.html) | [demo](09-bert-mlm/demo.py) |
| 10 · 生成式预训练 | GPT Causal LM | [阅读](10-gpt-causal-lm/README.md) | [打开](10-gpt-causal-lm/index.html) | [demo](10-gpt-causal-lm/demo.py) |
| 11 · Text-to-Text | T5 Span Corruption | [阅读](11-t5-span-corruption/README.md) | [打开](11-t5-span-corruption/index.html) | [demo](11-t5-span-corruption/demo.py) |
| 12 · 偏好优化 | SFT / DPO | [阅读](12-sft-dpo-alignment/README.md) | [打开](12-sft-dpo-alignment/index.html) | [demo](12-sft-dpo-alignment/demo.py) |
| 13 · 检索增强 | RAG | [阅读](13-rag-retrieval/README.md) | [打开](13-rag-retrieval/index.html) | [demo](13-rag-retrieval/demo.py) |
| 14 · 推理强化学习 | Reasoning RL / Verifier | [阅读](14-reasoning-rl-verifier/README.md) | [打开](14-reasoning-rl-verifier/index.html) | [demo](14-reasoning-rl-verifier/demo.py) |
| 15 · 数据工程 | LLM 数据清洗 / MinHash | [阅读](15-data-cleaning-minhash/README.md) | [打开](15-data-cleaning-minhash/index.html) | [demo](15-data-cleaning-minhash/demo.py) |
| 16 · 参数高效微调 | LoRA | [阅读](16-lora-finetuning/README.md) | [打开](16-lora-finetuning/index.html) | [demo](16-lora-finetuning/demo.py) |
| 17 · 计算与数据配比 | Scaling Laws | [阅读](17-scaling-laws/README.md) | [打开](17-scaling-laws/index.html) | [demo](17-scaling-laws/demo.py) |

## 每章怎么读？

1. 阅读 **README.md**：历史瓶颈、核心公式、为何下一代方法出现。
2. 用 **index.html** 手动切换算法步骤，并调节微实验参数，观察核心数值和掩码。
3. 跑 **demo.py**：先找到输入数据，再跟踪 forward、loss、decode。
4. 修改 toy data 或训练步数，预测改变并实际验证，最后运行测试。

## 数据集：按 nanoGPT 的方式按需下载

默认 toy 样本非常小，不追求基准分数。需要规模化时先读取 [datasets.json](datasets.json) 中的 dataset card、config、split、license，再独立下载小子集：

```bash
pip install -r requirements-data.txt
python scripts/prepare_data.py --lesson 10-gpt-causal-lm --limit 200
python scripts/prepare_data.py --lesson 14-reasoning-rl-verifier --limit 100
python scripts/prepare_shakespeare.py --limit-chars 50000
python 10-gpt-causal-lm/demo.py --steps 50 --text-file data/tiny_shakespeare_sample.txt
python 06-rnn-lstm/demo.py --steps 50 --text-file data/tiny_shakespeare_sample.txt
```

上述命令只**准备数据**，不会悄悄启动大型训练或把外部数据直接塞进所有 demo；进阶读者根据每章字段和任务改写 toy loader。统一保存为 `data/<chapter>/samples.jsonl`；Tiny Shakespeare 另存为 `data/tiny_shakespeare_sample.txt`。

推荐数据源：[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)、[Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare)、[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003)、[SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)、[OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)、[SQuAD](https://huggingface.co/datasets/rajpurkar/squad)、[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)。**Alpaca 为非商业许可；其他数据使用前请查各自卡片。**

## 项目构成

```text
nano-nlp/
├── 01-symbolic-grammar/ ... 17-scaling-laws/
│   ├── README.md    # 论文脉络、模型动机、公式和边界
│   ├── data.py      # 本章数据和任务定义
│   ├── model.py     # 可审查的核心算法
│   ├── project.py   # 训练/验证/保存逻辑
│   ├── inference.py # PyTorch 章节：独立恢复推理
│   ├── demo.py      # CLI 入口和 Loguru 记录
│   ├── visualize.js # 专属机制交互
│   └── index.html   # 交互教学与真实实验查看器
├── utils/
│   ├── retrieval.py
│   ├── viz.js
│   └── viz.css
├── scripts/
│   ├── prepare_data.py
│   └── prepare_shakespeare.py
├── tests/test_curriculum.py
├── datasets.json
├── PAPERS.md
├── AGENTS.md
└── .github/workflows/tests.yml
```

## 设计原则与诚实边界

- **先理解算子，再学习框架。** 基础阶段只用标准库；Torch 章节不引入 Lightning/Trainer/分布式训练，保留显式梯度更新。
- **不以“训练成功”代替“学懂原理”。** 例如 14 是 group-normalized REINFORCE，不是完整 GRPO；13 演示 RAG 的检索与上下文构造，不谎称已调用真实 LLM；11 不是原版完整 T5。
- **实际论文 PDF**：优先外链出版方/原作者文件。为避免向仓库存放大量第三方二进制及版权不确定文件，当前未直接提交论文 PDF；参见 [PAPERS.md](PAPERS.md)。
- **代码注释优先解释条件和边界。** 特别标注 teacher forcing、未来 token mask、group reward、质量过滤误杀、source provenance 等。
- **测试以数学不变量为先。** 默认不联网；神经章节运行 2 step smoke test（不是精度评测）。

## 参与开发

阅读 [STYLE.md](STYLE.md) 了解 nanoGPT 风格来源；阅读 [AGENTS.md](AGENTS.md) 查看教学标准、文件命名、可靠性和版权规范。欢迎按 18-xxx-xxx 继续扩展 CRF、GloVe、ELMo、RoBERTa、Contrastive Learning、Reranker、Mixture-of-Experts 与长上下文。

## 致谢

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy（参考简洁的训练与数据准备风格；本仓库算法代码独立编写）
- [Hugging Face Datasets](https://huggingface.co/datasets) 与各原始论文作者；本项目不将外部数据集/论文版权转移给教材。

## v2.0 · 17 章可恢复研究 MVP

v2.0 不再只提供小型单文件示例。每章现在都可在**训练/拟合进程退出之后**，从独立 checkpoint 恢复模型状态完成预测，并且有明确的训练、验证、数据源和算法边界。

```bash
git checkout v2.0
python -m pip install -r requirements.txt
# A. 训练与保存：默认 toy 数据，无需访问网络
python 10-gpt-causal-lm/demo.py --steps 30 --output-dir outputs/10-gpt-causal-lm
# B. 新进程加载同一 checkpoint 并生成
python 10-gpt-causal-lm/demo.py --mode predict --output-dir outputs/10-gpt-causal-lm --input 'the'
# C. 用 temperature/top-k 做可复现采样（只支持 GPT）
python 10-gpt-causal-lm/demo.py --mode predict --output-dir outputs/10-gpt-causal-lm --input 'the' --temperature 0.8 --top-k 5 --new-tokens 40
# D. 为所有章节提供外部小数据 schema
python 13-rag-retrieval/demo.py --text-file fixtures/13-rag-retrieval.jsonl --output-dir outputs/rag-local
python 13-rag-retrieval/demo.py --mode predict --output-dir outputs/rag-local --input 'which predicts masked tokens'
# E. 全部测试；包含跨进程模型恢复、17个数据 fixture 与数学不变量
python -m unittest discover -s tests -v
```

**统一产物：** `metrics.json`（评估指标与 seed）、`examples.jsonl`（验证样例）、`run.log`（Loguru）、`history.jsonl`（真实梯度优化轨迹；无梯度模型为空）和 `checkpoint.json` / `checkpoint.pt`（可恢复状态）。PyTorch 只存 `state_dict` 和构造参数，读取时使用 `weights_only=True`。

**交互实验：** 每章 `index.html` 既有独立的算法机制组件，也有无需后端的 Evidence Viewer。打开页面后从同章 outputs 导入 `metrics.json`、`history.jsonl` 和 `examples.jsonl`，浏览器显示**真实训练**曲线与预测，和页面上的公式示意严格区分。

**数据来源和许可证：** 17 个离线 fixture 的字段说明见 [fixtures/README.md](fixtures/README.md)。下载到本地的原始 Hugging Face 样本并不保证符合每章监督标签 schema；需要用户进行合法合规的数据转换，不能把纯文本当成已标注 benchmark。训练文本可以直接读取纯 TXT 或 `text` 字段 JSONL。

**代码导航：** 每章 `data.py → model.py → project.py → demo.py`；十个 PyTorch 章节还含 `inference.py`；全局 `utils/v2io.py`、`utils/runtime.py`、`utils/data.py` 保证统一安全工件、loguru 日志和输入校验。

- [v2.0 改造路线与严格验收](V2_ROADMAP.md)
- [贡献规则与注释标准](AGENTS.md)
- [论文链接索引](PAPERS.md)
- [nanoGPT 代码风格的教学取舍](STYLE.md)

**范围说明：** 完整教学型 MVP ≠ 完整工业框架或论文复现实验。RAG 采用抽取式证据回答而非外部 LLM 生成；推理 RL 是验证奖励的简化 policy gradient；Scaling Laws 的损失曲线是假设函数；真实 benchmark、复杂数据清洗与工业训练规模未被这些 toy 指标证明。

