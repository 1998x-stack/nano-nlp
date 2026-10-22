# nanoGPT 源码风格：nano-nlp 的设计基准

基于 [nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)、[nanoGPT/train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py) 与 [Tiny Shakespeare prepare.py](https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py) 阅读整理。原项目由 Andrej Karpathy 发布，MIT License。本教材没有整段复制源码。

| nanoGPT 取舍 | 本教材实现 | 教学目的 |
|---|---|---|
| 单文件直接表达模型结构 | 每章 `demo.py` 小模型 / 小算法 | forward 一眼可见，不依赖庞大框架 |
| 数据准备独立 | `scripts/prepare_data.py` / `scripts/prepare_shakespeare.py` | 下载、处理、训练分离 |
| 显式训练循环 | `optimizer.zero_grad → loss.backward → step` | 明确梯度在哪里产生 |
| 数学与张量对应 | qkv、mask、teacher forcing 旁有注释 | 每个关键计算有代码锚点 |
| 不过度封装 | 基础算法只用 Python 标准库 | NLP 历史不被 Torch 绑定 |
| 默认小型可运行实验 | toy 数据可离线运行，按需下载真实语料 | 先理解机制，后做实验规模化 |

**刻意不包含：** DDP、多节点训练、FlashAttention、可扩展 checkpoint 系统、大规模评测。这些很重要，但不适合在解释一个核心技术时与数学机理混在同一个最小实验。

关键判断标准：读者能否仅看当前文件就回答“输入是什么、哪一步定义条件独立性或 mask、loss 对什么求导、什么反例会破坏这个方法”。