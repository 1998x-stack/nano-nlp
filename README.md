# nano-nlp — 从规则 NLP 到 LLM 的可运行演进教材

> **学习目标：不仅知道模型名字，还能用最小代码重现它解决的关键问题。**
>
> 灵感来自 [Andrej Karpathy 的 nanoGPT](https://github.com/karpathy/nanoGPT)：优先透明代码、可复现 toy demo、独立数据准备、可阅读的训练循环。这里是教学实现，不是 nanoGPT 的复制品，也不保证生产级性能。

## 30 秒开始

```bash
git clone https://github.com/1998x-stack/nano-nlp.git
cd nano-nlp
python 01-symbolic-grammar/demo.py      # 零依赖
python 04-tfidf-bm25-search/demo.py    # 零依赖
python -m pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 25
python -m unittest discover -s tests -v
python -m http.server 8000             # 浏览器访问 /10-gpt-causal-lm/index.html
```

Python 3.10+。Torch 仅对神经网络章节必要；基础统计章节没有第三方依赖。**所有章节默认离线运行，无需 GPU、API Key 或下载大型模型。**

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
│   ├── demo.py      # 最小可运行算法与中文注释
│   └── index.html   # 可交互过程与参数微实验
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
