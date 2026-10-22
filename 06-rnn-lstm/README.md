# 06 · RNN 与 LSTM 序列语言模型

**时代：** 1997 / 2014

**问题：** 如何让当前词的预测依赖此前的信息，而不再局限于 N-gram？

## 最小模型的核心逻辑

Embedding → LSTM(hidden,cell) → Linear → Next-token Loss；门结构调节记忆，仍然逐时间步递归。

```text
(h_t,c_t) = LSTM(x_t, h_(t−1), c_(t−1))
```

## 逐行阅读指南

1. **词表与序列**：将每个字符映射到整数，构造输入 x 与右移标签 y；在代码中搜索 `ids[:-1], ids[1:]`。
2. **嵌入**：索引向量比 one-hot 更紧凑且可训练；在代码中搜索 `nn.Embedding(vocab, 24)`。
3. **门控记忆**：LSTM 维护隐藏状态 h 与记忆状态 c；在代码中搜索 `self.rnn(self.embed(x))`。
4. **预测训练**：使用每个时间步的 logits 计算交叉熵；在代码中搜索 `F.cross_entropy(logits, target)`。

## 运行

```bash
pip install -r requirements.txt
python 06-rnn-lstm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://www.bioinf.jku.at/publications/older/2604.pdf

## 重要边界

LSTM 改善梯度流动却没有消除沿时间的串行计算。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 06-rnn-lstm/demo.py --steps 30 --output-dir outputs/06-rnn-lstm
cat outputs/06-rnn-lstm/metrics.json
```

**本章实际可验收任务：** 字符级语言建模；独立验证段 NLL、PPL 与不带标签的生成。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 字符级输入按时间展开，LSTM 的 f、i、o 门管理 cell state；每个位置输出下一字符词表概率。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 06-rnn-lstm/demo.py --steps 30 --output-dir outputs/06-rnn-lstm
python 06-rnn-lstm/demo.py --mode predict --output-dir outputs/06-rnn-lstm --input 't'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_nll、val_perplexity、逐步训练 loss 和 greedy 生成的字符。

**关键正确性断言：** 验证划分按文本尾部分开；标签必须比输入右移一位且不能读取未来字符。

**建议的对照实验：** 对照字符 bigram 的 PPL，改变 block_size 和训练步数观察变化。

**能力边界：** 此 MVP 的每次生成重放短序列，没有保留跨窗口的 KV cache 或 LSTM hidden cache。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？


## v3.0 · 验证词表与 NLL 纠偏

v2 在全量语料上建立字符表，且验证损失使用固定窗口简单平均：可能看到未来验证集才有的字符，也会漏掉最后不足一个 block 的目标字符。

v3 使用 [utils/lm_eval.py](../utils/lm_eval.py) 完成：

1. **先切分文本**，再只从训练段构建 `[<unk>, sorted(train_chars)]`。
2. 验证未见字符 ID 固定映射为 0，并记录 `val_oov_tokens`。
3. 遍历验证段的每个目标字符（最后一段不足 `BLOCK` 也纳入）。
4. 计算 `mean NLL = Σ_i (window_loss_i × window_targets_i) / Σ_i window_targets_i`，其中 `val_evaluated_tokens=len(validation_ids)-1`。
5. `val_perplexity=exp(mean NLL)`；模型的词表、unk_id 和窗口长度一同保存。

命令：`python 06-rnn-lstm/demo.py --steps 30 --output-dir outputs/lstm-v3`。之后可在独立进程恢复并生成；通过 `python 06-rnn-lstm/demo.py --mode inspect --output-dir outputs/lstm-v3` 核验实验血缘。

**注意：** 验证文本与训练文本若具有高度重复的模板，仍可能高估真实语言泛化。这里只纠正计算协议，未宣称 Toy 语料构成严谨公开基准。
