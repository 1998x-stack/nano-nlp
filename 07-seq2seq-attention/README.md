# 07 · Seq2Seq 与 Bahdanau 风格 Attention

**时代：** 2014–2015

**问题：** 为什么翻译长句时，Decoder 不该只依赖单个固定长度向量？

## 最小模型的核心逻辑

Encoder 提供每个位置的隐藏状态，Decoder 在每一步动态计算注意力权重，并加权组合输入上下文。

```text
α_t=softmax(score(h_dec,h_enc_i)); c_t=Σ α_(t,i) h_enc_i
```

## 逐行阅读指南

1. **编码源句**：GRU 输出全部 encoder hidden states；在代码中搜索 `memory, hidden = encoder(source)`。
2. **计算对齐**：Decoder hidden 与源端各位置相似度；在代码中搜索 `scores = memory @ hidden`。
3. **动态上下文**：softmax 得到权重并求加权和；在代码中搜索 `context = attention @ memory`。
4. **自回归解码**：预测目标词，teacher forcing 提供下一个输入；在代码中搜索 `target[:, t]`。

## 运行

```bash
pip install -r requirements.txt
python 07-seq2seq-attention/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/Helsinki-NLP/opus_books
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1409.0473

## 重要边界

这是教学级 dot-product cross-attention，而非 Bahdanau 论文中的 additive score；两者都缓解固定向量瓶颈。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 07-seq2seq-attention/demo.py --steps 30 --output-dir outputs/07-seq2seq-attention
cat outputs/07-seq2seq-attention/metrics.json
```

**本章实际可验收任务：** 词级 Encoder–Decoder 小翻译；Teacher Forcing 训练，但评估时必须自回归解码；输出逐词注意力。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 编码器保留每个 source 位置的表示；解码器每步由 query·memory 得到对齐权重，再利用真实 target 做 teacher forcing。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 07-seq2seq-attention/demo.py --steps 30 --output-dir outputs/07-seq2seq-attention
python 07-seq2seq-attention/demo.py --mode predict --output-dir outputs/07-seq2seq-attention --input '1,2,3'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_token_accuracy、val_exact_match、生成序列以及每步 source attention。

**关键正确性断言：** 推理阶段禁止使用真实 target；必须每一步喂回模型自己刚预测的 token。

**建议的对照实验：** 将源序列顺序打乱，对照 attention 权重和预测，确认当前解码步使用了源端不同信息。

**能力边界：** 示例是数字 token 玩具翻译、点积 cross attention，不是论文中的 additive attention。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
