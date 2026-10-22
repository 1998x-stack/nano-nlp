# 17 · Scaling Laws 与数据计算配比

2020–2022 · **要解决的问题**：模型参数与数据 tokens 都能提升能力，但固定算力应该怎么分配？

## 一句话理解

用教学化的幂律损失表达式模拟参数规模与 token 数量对最终 loss 的边际收益；在 N·D 预算限制下扫描候选组合。

```text
L(N,D) = E + A/N^α + B/D^β; C ∝ N·D
```

## 代码阅读路线

1. **固定计算量**：用 N×D 近似 Transformer 训练 FLOPs 的规模关系。代码锚点：`budget`。
2. **设定假设损失**：模型不足和数据不足分别带来幂律项。代码锚点：`loss_model(n,d)`。
3. **搜索候选**：改变 N/D 并淘汰超预算候选。代码锚点：`if n*d <= budget`。
4. **比较最低点**：找模拟条件下的 loss 最低组合。代码锚点：`min(candidates, key=...)`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 17-scaling-laws/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/Salesforce/wikitext

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2203.15556

## 局限与误区

演示参数是手工构造的假设曲线，不是从 Chinchilla 或 GPT-3 真实实验拟合；切勿据此报告最优训练规模。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 17-scaling-laws/demo.py --steps 1 --output-dir outputs/17-scaling-laws
cat outputs/17-scaling-laws/metrics.json
```

**本章 MVP 的验收任务：** 枚举 compute-constrained 可行 (N,D)，输出多个预算下的损失最小组合、实际预算占用和规则校验。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。
