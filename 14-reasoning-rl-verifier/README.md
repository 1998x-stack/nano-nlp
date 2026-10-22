# 14 · 推理 RL：可验证奖励与组内优势

2025 · **要解决的问题**：如何在没有逐 token 标准推理过程时，通过正确性反馈训练策略？

## 一句话理解

Policy 采样多个候选 → Verifier 检查最终答案 → 组内标准化 reward → policy gradient 更新。

```text
A_i = (r_i − mean(r_group)) / (std(r_group)+ε)
```

## 代码阅读路线

1. **采样操作**：对于 + 或 × 指令，模型采样候选操作。代码锚点：`Categorical(logits=...)`。
2. **外部验证**：真正执行整数计算，检查候选结果是否正确。代码锚点：`verify(op, a, b, answer)`。
3. **组内归一化**：把组内 reward 变成相对优势。代码锚点：`(reward-mean)/(std+eps)`。
4. **策略更新**：优先提高获得正优势的采样概率。代码锚点：`-(adv * log_prob).mean()`。
5. **检查泛化**：检查两种操作任务的最终策略。代码锚点：`policy.argmax(-1)`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 14-reasoning-rl-verifier/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/openai/gsm8k

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://arxiv.org/pdf/2501.12948

## 局限与误区

本章是 group-normalized REINFORCE 教学示例，并非完整 GRPO：未包含 PPO clipping、KL 控制、长轨迹语言策略。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 完整可运行的任务闭环

模块分工：`data.py` 定义输入与验证样本，`model.py` 对应论文数学或可学习结构，`project.py` 负责估计/训练/独立评估/推理并提供错误切片，`demo.py` 统一日志、参数和 JSONL 工件。

```bash
pip install -r requirements.txt
python 14-reasoning-rl-verifier/demo.py --steps 30 --output-dir outputs/14-reasoning-rl-verifier
cat outputs/14-reasoning-rl-verifier/metrics.json
```

**本章 MVP 的验收任务：** 采样分组候选答案，以外部确定性 verifier 评分并更新策略，在未训练数值上检查成功率。

**误差分析方向：** 不把 toy 结果外推为工业训练能力；保留每条 examples.jsonl 并改变语料/阈值/随机种子，观察为什么结论会改变。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 从同一个算术问题采样多个候选运算，以确定性 verifier 核对最终结果，然后组内归一化 reward 得到 policy gradient。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → `inference.py` 恢复权重后执行预测 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 14-reasoning-rl-verifier/demo.py --steps 30 --output-dir outputs/14-reasoning-rl-verifier
python 14-reasoning-rl-verifier/demo.py --mode predict --output-dir outputs/14-reasoning-rl-verifier --input 'add,3,4'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** val_success_rate、last_group_reward、采样概率、可验证执行结果。

**最值得做的消融实验：** 令同组候选全部正确或全部错误，证明 advantage 无法提供有效梯度。

**不要过度解读：** 只教学 group-normalized REINFORCE；未实现完整 GRPO 的 clipping、KL 和长轨迹。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
