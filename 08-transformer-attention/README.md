# 08 · Transformer Self-Attention 机制

**时代：** 2017

**问题：** 如何去掉 RNN，使任意两个 token 可以直接交互并进行并行计算？

## 最小模型的核心逻辑

Q/K/V 线性投影、缩放点积、Softmax、因果掩码、多头拼接，是 Decoder Transformer 的计算内核。

```text
softmax(Q Kᵀ / √d_k + causal_mask) V
```

## 逐行阅读指南

1. **生成 QKV**：共享输入通过三组线性投影形成 query/key/value；在代码中搜索 `self.qkv(x).chunk(3, dim=-1)`。
2. **拆分多头**：把通道拆为 heads × head_dim；在代码中搜索 `reshape(B, T, H, D).transpose(1,2)`。
3. **注意力分数**：点积并除 sqrt(head_dim) 防止规模过大；在代码中搜索 `(q @ k.transpose(-2,-1))/sqrt(D)`。
4. **因果掩码**：禁止当前位置看未来 token；在代码中搜索 `masked_fill(mask, -inf)`。
5. **加权汇总**：softmax 后乘以 V，并拼接各头；在代码中搜索 `weights @ value`。

## 运行

```bash
pip install -r requirements.txt
python 08-transformer-attention/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://arxiv.org/pdf/1706.03762

## 重要边界

Self-Attention 本身不包含位置概念；需要显式位置编码或其他位置机制。本章专注 MHA 算子。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 08-transformer-attention/demo.py --steps 30 --output-dir outputs/08-transformer-attention
cat outputs/08-transformer-attention/metrics.json
```

**本章实际可验收任务：** 使用 causal multi-head attention 在 toy 分类上训练/验证，并保存末位置 attention 权重。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** Q、K、V 投影后分别重排为 [B,H,T,D]，以 1/√D 缩放并掩盖未来 token；末尾 token 汇聚全序列做分类。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 08-transformer-attention/demo.py --steps 30 --output-dir outputs/08-transformer-attention
python 08-transformer-attention/demo.py --mode predict --output-dir outputs/08-transformer-attention --input '1,2,3,0'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_accuracy、注意力矩阵形状、右上角未来位置严格为 0。

**关键正确性断言：** 推理导出每个 head 的概率权重，可借助 HTML 查看下三角可见性。

**建议的对照实验：** 改变未来位置的 embedding，验证前面 token 的注意力输出保持不变，作为因果不泄漏测试。

**能力边界：** 这个项目聚焦 attention 算子，不等于完整的可训练 Decoder Transformer；完整 GPT 见第 10 章。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
