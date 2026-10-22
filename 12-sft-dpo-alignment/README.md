# 12 · SFT 与 DPO 偏好优化

2022–2023 · **要解决的问题**：语言模型会预测词，为什么还需要专门训练‘更符合人类偏好’的回答？

## 一句话理解

SFT 最大化正确回答概率；DPO 优化 chosen/rejected 相对参考模型的对数似然差。

```text
−log σ(β[(log πθ(y+)/πref(y+))−(log πθ(y−)/πref(y−))])
```

## 代码阅读路线

1. **构造偏好对**：为同一输入指定 chosen 与 rejected。代码锚点：`CHOSEN, REJECTED`。
2. **SFT 目标**：增加标准回答的条件似然。代码锚点：`F.cross_entropy(...)`。
3. **冻结参考模型**：参考概率不应随策略更新。代码锚点：`reference.eval()`。
4. **DPO 奖励差**：策略与参考的 log 概率变化比较。代码锚点：`gap=(policy_delta-reference_delta)`。
5. **偏好损失**：logsigmoid(beta * gap) 提升偏好相对优势。代码锚点：`-F.logsigmoid(beta*gap)`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 12-sft-dpo-alignment/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/tatsu-lab/alpaca

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2305.18290

## 局限与误区

只演示单 token 回答的偏好公式，不能把它理解为完整聊天模型 SFT/DPO 实现；Alpaca 数据许可为 CC BY-NC 4.0。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 12-sft-dpo-alignment/demo.py --steps 30 --output-dir outputs/12-sft-dpo-alignment
cat outputs/12-sft-dpo-alignment/metrics.json
```

**本章 MVP 的验收任务：** 区分 SFT 的示范似然与 DPO 的成对偏好目标；冻结参考模型；报告验证集 chosen 胜率和偏好优势变化。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 先用 chosen 做 SFT，然后冻结当前策略作为 reference，再用 DPO 提升 chosen/rejected 对数优势。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → `inference.py` 恢复权重后执行预测 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 12-sft-dpo-alignment/demo.py --steps 30 --output-dir outputs/12-sft-dpo-alignment
python 12-sft-dpo-alignment/demo.py --mode predict --output-dir outputs/12-sft-dpo-alignment --input '0'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** val_chosen_win_rate、mean_log_margin_improvement、reference_frozen 和独立推理概率。

**最值得做的消融实验：** 将 DPO beta 增大/减小，观察 chosen 相对于 reference 的 log margin 变化。

**不要过度解读：** 分类 token 是教学极简数据，不能宣称提供真实聊天安全对齐。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
