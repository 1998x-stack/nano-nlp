# nano-nlp v2.0 改造计划与验收标准

## 目标：从 17 个算法片段升级为可运行的完整教学 MVP

每章必须形成以下闭环：

**明确任务与成功标准 → 准备/校验数据 → 初始化/训练模型（或统计估计）→ 独立验证 → 推理 → 保存指标与样本结果 → 交互式可视化分析**。

与 nanoGPT 一样，核心代码应尽量直白；与 nanoGPT 不同，早期 NLP 的算法不应为了形式统一而被错误封装为神经网络。

## 代码布局（每章）

- \`data.py\`：训练/验证样本、预处理、数据集校验与 task-specific tokenizer；不能在 import 时联网。
- \`model.py\`：算法、数学实现、神经网络 forward、概率及推断机制；保留关键公式与 tensor shape 注释。
- \`project.py\`：本章独特的训练、评估、推理全流程和样本工件；使用 loguru 记录阶段信息。
- \`demo.py\`：薄 CLI 入口，支持 \`--steps\`（需要训练时）、\`--output-dir\`、\`--seed\`，最后打印 \`RESULT:\`。
- \`README.md\`：问题动机→数据字典→公式手算→模块关系→训练评估→实验对比→失败案例→论文。
- \`index.html\` + 专属 \`visualize.js\`：与本章数学一致的交互，不能仅重用同一流程图换标题。

共享目录：
- \`utils/runtime.py\`：seed、loguru、输出工件、稳定 JSON。
- \`utils/data.py\`：简易 JSONL / 文字读取与划分。
- \`utils/metrics.py\`：可复用的分类、检索、概率指标。
- \`scripts/prepare_data.py\`：可选外部下载，不影响任何章默认运行。

## 17 章最终实验范围

| 阶段 | 可执行任务 | 训练/估计 | 独立验证和推理 | 最重要的可视化 |
|---|---|---|---|---|
| 01 CFG | 语法可接受性 + 句法树 | 明确产生式与词典 | held-out 合法/非法句解析 | 展开树与规则匹配 |
| 02 N-gram | 下词预测与文本生成 | 估计频数/平滑 | held-out NLL、困惑度及预测 | α 对词概率影响 |
| 03 HMM | POS 序列标注 | 频数估计 START/TRANS/EMIT | Viterbi、token accuracy | Viterbi 网格和回溯 |
| 04 BM25 | 文档检索 | 建倒排与 IDF | Recall@k / MRR / top-k | 分解 TF/IDF/长度 |
| 05 SkipGram | 词向量学习 | 正/负采样 SGD | 正负分数与最近邻 | 负采样概率与相似度 |
| 06 LSTM | 语言建模 | BPTT | validation loss、greedy sample | 门控记忆与序列 |
| 07 Seq2Seq | 条件翻译 | teacher forcing | exact sequence accuracy、decode | 对齐/交叉注意力 |
| 08 Transformer | 文本分类 | 自注意力 Encoder + Head | accuracy / attribution demo | attention 矩阵 / mask |
| 09 BERT MLM | 掩码补全 | 双向 Encoder | masked-token accuracy | 双向 mask 与位置 |
| 10 GPT | 字符级语言生成 | Decoder causal LM | validation NLL、sample | QKV、因果矩阵 |
| 11 T5 | span corruption | Encoder–Decoder | sentinel 片段重建 | span/sentinel 映射 |
| 12 SFT/DPO | 双候选偏好选择 | NLL + DPO | chosen-win accuracy | log-ratio / β |
| 13 RAG | 证据问答 | 索引 + query | evidence hit、extractive answer | 检索 Top-k 与引用 |
| 14 Reasoning RL | 可验证计算动作 | group policy gradient | verifier 成功率 | group advantage |
| 15 Dedup | 数据清洗与去重 | 文档签名与 LSH 桶 | 原始/去重比例与候选命中 | Jaccard / LSH 曲线 |
| 16 LoRA | PEFT 回归适配 | 冻结基座 + 低秩优化 | held-out MSE、参数对比 | rank / trainable fraction |
| 17 Scaling | 预算/损失建模 | 幂律拟合或扫描 | budget feasible candidates | N/D loss surface |

## 统一执行契约

\`\`\`bash
pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 30 --output-dir outputs/gpt-v2
python -m unittest discover -s tests -v
\`\`\`

每章输出：
- 控制台有 \`RESULT: {...}\` 可机读摘要。
- \`outputs/<chapter>/metrics.json\`：任务指标、随机种子、配置、样本数。
- \`outputs/<chapter>/examples.jsonl\`：可观察的真值、模型预测、证据或中间分数。
- \`outputs/<chapter>/run.log\`：loguru 的阶段化日志。
- 若神经模型保存小权重，明确声明示例权重不是生产模型。

## 测试分层

1. **数学不变量**：概率和为 1、因果 mask 的未来位置为 0、Viterbi 回溯、DPO 参考冻结、输出引用可追踪。
2. **任务集成**：一条命令能做训练→验证→推断→落盘，指标结构正确。
3. **视觉测试**：每章 HTML 能单独打开、可调节参数、显示输入和输出。
4. **CI**：无联网、CPU、固定 seed；失败不得假装完成。

## 文档原则

- 所有真实数据 URL + config/split/field/许可写入 \`datasets.json\`；数据下载是 opt-in。
- 所有论文保留出版社或作者原始地址，PDF 在取得适当许可时才存仓库。
- 准确声明教学简化：例如 RL 章节不是完整 GRPO；RAG 不能把检索模块当生成器。
- \`loguru\` 对于完整项目是必要依赖，不在每个模块里重复配置 sink；统一在 CLI 初始化。
- 模块 import 不能训练、下载或写文件；防止意外副作用。

## 改造顺序

- **阶段 A**：共享 runtime/data/metrics，模块组织约定、v2.0 CI。
- **阶段 B**：01–05 规则/统计/词向量完整训练与评估。
- **阶段 C**：06–11 神经模型数据流、优化、独立验证与生成。
- **阶段 D**：12–17 后训练/检索/数据工程/计算预算。
- **阶段 E**：专属交互、README 深化、统一集成测试和课程审查。

> 进度应依据真实提交与自动测试更新。不能用“目录已创建”代替“完整 MVP 已验证”。
