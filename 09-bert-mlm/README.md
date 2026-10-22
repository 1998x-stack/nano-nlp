# 09 · BERT 双向表示与 Masked LM

**时代：** 2018–2019

**问题：** 为什么理解任务需要同时利用 token 左右两侧的上下文？

## 最小模型的核心逻辑

对随机隐藏的 token 做分类；Encoder 无因果掩码，隐藏位置可以看见左右未遮盖 token。

```text
L_MLM = −Σ_(i∈masked) log P(x_i | x_masked)
```

## 逐行阅读指南

1. **构造 Mask**：选位置替换为 [MASK]，仅这些位置参与 loss；在代码中搜索 `inputs[mask_positions] = MASK`。
2. **双向 Encoder**：self-attention 允许读取左右上下文；在代码中搜索 `nn.TransformerEncoder`。
3. **预测原词**：为 masked token 位置输出词表 logits；在代码中搜索 `head(hidden)`。
4. **稀疏损失**：计算原始目标词的 cross entropy；在代码中搜索 `logits[mask_positions]`。

## 运行

```bash
pip install -r requirements.txt
python 09-bert-mlm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/stanfordnlp/sst2
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1810.04805

## 重要边界

本章只教学 MLM，不包含原始 BERT 的 NSP、WordPiece 词表、80/10/10 mask 策略及大型预训练。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 09-bert-mlm/demo.py --steps 30 --output-dir outputs/09-bert-mlm
cat outputs/09-bert-mlm/metrics.json
```

**本章实际可验收任务：** 双向 Masked LM；训练只对 mask 位置计算 CE，验证单独逐位置 mask 并计算准确率。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 仅对显式 [MASK] 位置计算 cross entropy；Encoder 可读取左右两个方向的可见 token。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 09-bert-mlm/demo.py --steps 30 --output-dir outputs/09-bert-mlm
python 09-bert-mlm/demo.py --mode predict --output-dir outputs/09-bert-mlm --input '1,0,3,4'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_mask_accuracy、masked_position、gold 与 prediction 的逐 token 对照。

**关键正确性断言：** 独立恢复后如果输入没有 0=[MASK] 必须拒绝，而不是把分类结果当生成结果。

**建议的对照实验：** 单独遮盖句首、句中、句末，观察左右上下文依赖；尝试去掉 position embedding。

**能力边界：** 仅实现 MLM 核心，没有复现原论文完整 NSP、WordPiece 与大规模预训练。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
