# 16 · LoRA 参数高效微调

2021–2023 · **要解决的问题**：微调 LLM 时，为什么可以只训练极少量的低秩参数？

## 一句话理解

冻结预训练权重 W，只学习低秩增量 BA 并通过缩放系数叠加到线性层输出。

```text
y = Wx + (α/r)BAx
```

## 代码阅读路线

1. **冻结基座**：W 的 requires_grad=False，避免全参数更新。代码锚点：`self.base.weight.requires_grad_(False)`。
2. **低秩矩阵**：训练 A: in→rank，B: rank→out。代码锚点：`self.A, self.B`。
3. **前向叠加**：base(x) + B(A(x))*alpha/r。代码锚点：`self.base(x) + self.B(self.A(x))*...`。
4. **只优化 adapter**：优化器仅接收 A、B 参数。代码锚点：`optimizer = AdamW([*lora.A.parameters(), ...])`。
5. **统计可训练比**：对比可训练参数与总参数。代码锚点：`trainable, total`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 16-lora-finetuning/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/stanfordnlp/sst2

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2106.09685

## 局限与误区

真正的 LoRA 微调需要选择注意力/FFN 投影、考虑 rank 与任务、保存 adapter，并与量化和 tokenizer 配合。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 16-lora-finetuning/demo.py --steps 30 --output-dir outputs/16-lora-finetuning
cat outputs/16-lora-finetuning/metrics.json
```

**本章 MVP 的验收任务：** 冻结基座权重并仅更新 LoRA Adapter；独立验证集 MSE、参数量和权重保持不变的断言。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 冻结基础矩阵 W，只训练低秩 A、B；对同一个输入比较原冻结基座与叠加 adapter 的输出。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → `inference.py` 恢复权重后执行预测 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 16-lora-finetuning/demo.py --steps 30 --output-dir outputs/16-lora-finetuning
python 16-lora-finetuning/demo.py --mode predict --output-dir outputs/16-lora-finetuning --input '1,0,0,0,0,0,0,0'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** baseline_val_mse、val_mse、trainable_fraction、base_unchanged、adapter_delta。

**最值得做的消融实验：** 改变 rank 与 alpha，对比可训练参数量和验证 MSE；检查 B 零初始化时初始行为与基座相同。

**不要过度解读：** 小矩阵上的参数节省比率不能代表 7B/70B 模型真实 PEFT 收益。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
