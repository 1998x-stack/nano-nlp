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
