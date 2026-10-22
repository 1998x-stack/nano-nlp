# nano-nlp v2.0 — 深度改造与验收路线

> 状态：持续迭代。基线：`v2.0` 提交 `266c360`（17 章模块化、跨章集成测试、loguru 日志与共享实验工件，GitHub Actions 已通过）。本文件是**交付计划与验收契约**，不能将规划内容误写为已实现。

## 目标与范围

把每个章节从概念 demo 升级为**可独立完成任务的教学型 MVP**：输入、拟合/训练、独立验证、推理、日志、可追踪工件、专属可视化与测试。保持 nanoGPT 风格——最少工程层数暴露最多核心算法，不将教学内容掩藏在通用训练框架里。

## 当前审查：已经完成的部分

- 17 章目前均有 `data.py`、`model.py`、`project.py`、`demo.py`、`README.md`、`index.html`、`visualize.js`。
- `utils/runtime.py` 统一日志入口、种子、CLI 和 `metrics.json`/`examples.jsonl`/`run.log`；`utils/data.py` 负责本地数据读取；`utils/metrics.py` 负责部分共用指标。
- CI 已运行跨章节 v2 MVP 测试：17 章以小步数验证可执行和结果工件；该测试**不能**替代模型泛化、真实数据适配、算法正确性的深入测试。
- 部分实验仍使用小型、固定或单任务数据（如 GPT 字符建模、SFT/DPO 二分类）、有限超参和特定于示例的验证逻辑；它们是教学原型，距离真实任务 MVP 的工程覆盖仍有差距。
- HTML 有逐步演示及数值交互；仍需核验是否真正可消费后端保存的训练曲线和结果，而不仅是独立假设模拟。

## 必须遵守的代码合同

```text
XX-topic/
├── README.md       # 问题背景、核心数学、张量形状、边界、复现实验
├── data.py         # Toy 数据、解析、训练/验证分割与无泄漏测试
├── model.py        # 独立算法实现（不得混入下载/终端命令）
├── project.py      # 组装训练/拟合、评估、推理和导出工件
├── demo.py         # 极小 CLI 入口，调用 utils.runtime.run_cli
├── index.html      # 本章专属交互教学体验
├── visualize.js    # 本章真实机制的动画/参数探究
└── tests/          # 每章至少不变量、回归和端到端检查（逐步迁移）
```

原则：每章允许增加 `train.py`、`evaluate.py`、`generate.py` 或 `config.py`，但只有当它们能明显降低认知负担才拆分。不能为了文件数量多而拆分。训练逻辑要明确可读；`project.py` 可以调用共享输入输出工具，但不能在抽象层中隐藏算法自身的 loss 或推断逻辑。

## v2.0 验收标准（必须实际测试）

1. **数据闭环**：默认 toy 数据离线可跑；支持本地真实小语料；训练与验证不重叠；注明 source/license/split，固定 seed。预测阶段遇到 OOV、空输入、短数据需有定义行为。
2. **算法闭环**：`fit/train → evaluate → predict` 真实成立；保存至少一个任务有效指标和典型案例；与论文关键公式对应；不把 toy loss 等同真实 benchmark。
3. **工件闭环**：`outputs/<chapter>/{run.log,metrics.json,examples.jsonl}` 可复核；需要复用模型的章节增加 checkpoint、vocab、超参、版本元数据，并提供从 checkpoint 恢复预测的入口。
4. **日志**：用 `loguru`，入口配置 sink/level，模块采用 `from loguru import logger`；报告数据量、运行设备、epoch/step、loss、验证指标和失败原因。避免 token 级无节制日志，绝不输出凭据和隐私。
5. **HTML**：至少一个可直接操作的本章专属机制；可视化使用正确计算公式和标明 Toy/真实结果；可选导入 `metrics.json`，展示 loss 和预测示例；不允许假造来自真实训练的数值。
6. **README**：动机与历史位置；数据例子；问题定义；关键公式；逐步代码定位；复杂度与限制；可复现实验命令；对照实验；常见错误；下一章为什么需要出现。
7. **测试**：算法单元与数学不变量、seed 复现、合法/非法输入、训练有效性、独立推理、工件 schema、HTML JS syntax、CPU CI。训练有效性通过小数据 overfit/损失下降等适当检查，而非所有模型统一固定 loss 阈值。
8. **性能边界**：基础版本 Python 3.10+、CPU 运行；神经模型 PyTorch + loguru；真实数据下载独立执行，不在测试时隐式联网。

## 按领域分组的改造任务

| Wave | 章节 | MVP 的核心验收 |
|---|---|---|
| A：无梯度经典 NLP | 01 CFG；02 Bigram；03 HMM；04 BM25 | 文法解析/生成与歧义；NLL/PPL/OOV；训练估计的 HMM + Viterbi；可重建索引与 Recall@K |
| B：学习表示与序列 | 05 Word2Vec；06 LSTM；07 Seq2Seq | 词向量训练和近邻；字符生成及 checkpoint；翻译 teacher forcing/自回归解码、注意力可视化 |
| C：预训练架构 | 08 Transformer；09 BERT；10 GPT；11 T5 | 模型输出与因果掩码验证；MLM 预测与下游任务；训练/采样/载入；span corruption/条件生成 |
| D：LLM 后训练 | 12 SFT-DPO；14 Reasoning RL；16 LoRA | 保存 policy/reference 对比、偏好数据校验；可执行 verifier、reward 曲线；adapter 保存、恢复和效果对比 |
| E：系统、数据与规律 | 13 RAG；15 MinHash；17 Scaling Laws | 索引/召回/引用/拒答；Exact/近似去重、误报分析；拟合实验数据而非仅绘制假设幂律 |

### 第一优先：10-gpt-causal-lm 完整样板

1. 将 `data.py` 增强为固定长度样本、训练/验证独立切分、字符 vocab 持久化，提供 Tiny Shakespeare 的本地路径入口。
2. `model.py` 保留显示 QKV、多头、causal mask、MLP、残差连接的清晰代码；单元测试未来 token 的改动不得改变之前位置的 logits。
3. `project.py` 增加 checkpoint 保存/加载、验证 PPL、采样温度/top-k、早期小样本过拟合检查、loss 曲线 JSONL 输出。
4. `demo.py` 支持 `train` / `eval` / `sample` 命令，同时保留兼容默认入口。
5. HTML 展示 token 编码、下三角 mask、各注意力头、loss 曲线与生成结果；明确标明显示的是模拟数据还是导入的实验数据。
6. README 给出逐层 shape 表、attention 手算、最小训练命令、推理恢复命令、训练后 checkpoint 路径、技术边界以及 nanoGPT 对照阅读章节。
7. 加入完整训练→保存→重新进程加载→生成 smoke test 与从未见数据中验证 PPL 的回归测试。

只有样板通过后再将共享的**工件协议与测试约定**推广到其他章节；不强迫它们采用 GPT 的特定模型接口。

## 执行阶段

- **P0 基线审计**：冻结已通过 CI 的 commit；检查现有 17 章模型与 README/HTML 一致性，记录需要修复的算法问题。
- **P1 规范与 GPT 样板**：统一配置、日志、产物结构；完成第 10 章端到端真实训练及 checkpoint 恢复、专属 HTML。
- **P2 经典 NLP 四章**：完成可拟合、可预测、可评估的统计/检索模块，增加数学不变量测试。
- **P3 神经模型五章**：训练和推理拆分，补齐基础可复现实验和解释性可视化。
- **P4 后训练及系统五章**：偏好、验证器、RAG、去重与 Scaling 的证据与评估。
- **P5 文档与质量验收**：逐章核对论文与技术简化、外部数据契约、实验报告、可视化数值与 README、CI 稳定性。

## 评审的拒绝条件

- 只是把原 demo 复制到 `model.py`，却没有独立拟合/预测/评估；
- 训练样本和验证样本混淆、因果模型未来标签泄漏；
- 只保存 metrics，没有真正的可恢复模型，仍声称“完整预训练 MVP”；
- HTML 展示了人为固定的动画或 loss 却标注成实际实验结果；
- 用 mock 或伪造数据输出代替真实算法；
- 以两步 smoke 成功宣称“算法在实际数据集上已被验证”。

## v2 深度改造实施清单（在最新 CI 验证后评审）

| 功能 | 覆盖范围 | 目前代码及可查验位置 |
|---|---|---|
| 分章节模型与真实任务闭环 | 17/17 | 每章 `data.py/model.py/project.py/demo.py` |
| 从持久状态独立恢复预测 | 17/17 | `project.predict_saved`；10 个 Torch 章节专有 `inference.py` |
| Loguru 与真正优化步骤历史 | 17/17（无梯度章节记录为空） | `utils/runtime.py`、`utils/v2io.py` |
| 本地文本及任务标签 JSONL | 17/17 | `fixtures/`、`utils/data.py`、各章 `project.py` |
| 浏览器导入训练指标和案例 | 17/17 | 每章 `index.html` 引用 `utils/experiment-viewer.js` |
| 跨进程 train/predict 单元与集成检查 | 17/17 | `tests/test_v2_restore.py`、`tests/test_v2_fixture_integration.py` |
| 数学不变量检查 | 神经架构与基础算法关键点 | `tests/test_math_invariants.py` 与基础单测 |
| 研究级外部 benchmark 性能复现 | 不属于本次教学 MVP | 原论文训练成本/大型数据与本项目不同 |
| 独立 checkpoint-only 重新评分命令 | 尚未普遍实现 | 现在训练时执行 held-out 验证；有跨进程 `--mode predict` |
| 任务级错误分析可视化 | 实验案例 JSONL 可以导入 | 更复杂的 confusion matrix、PR/ROC 不是所有任务的共同合理指标 |

这张表描述代码功能和可检查位置，**不等同于所有模型达到公开 benchmark 的论文成绩**。只有 GitHub CI 的对应提交全部通过后，才可确认这些运行性检查已被自动验证。需要进一步增加任务级数据集研究报告的章节，应单独立项，不将玩具评估冒充官方 benchmark。
