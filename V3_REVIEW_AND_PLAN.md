# nano-nlp v3.0 — 真实性、可复现性与评测协议

> 基于 [v2.0](https://github.com/1998x-stack/nano-nlp/tree/v2.0) 的逐文件审查（2026-10-22）。v2.0 最新提交 `b7fd549` CI 已通过，但 CI 通过表示程序可运行，不表示研究性评估正确或论文等价实现。
>
> **优先原则：修复评测与数学正确性 > 完善数据血缘 > 交互与可读性 > 新增技术主题。**

## 逐代码审查：已证实的问题

| 级别 | 章节/源码 | 问题 | 改造验收 |
|---|---|---|---|
| P0 | 02 `model.py`, `project.py` | 对不在输出词表的词直接做加一平滑，结果概率可以加到 >1；验证 OOV 只给告警 | 显式 `<unk>`，所有条件概率归一化；测试 OOV 与保存恢复一致 |
| P0 | 06/10 `project.py` | 训练/验证切分之后仍用**完整文本**构造字符表；验证可影响输入空间 | 先切分，再只由训练段建 vocab；验证 OOV 映射特殊 ID；记录 OOV 数量 |
| P0 | 06/10 `project.py` | 验证按 `range(0,len(val)-BLOCK-1,BLOCK)` 取固定长度块，会漏尾部 token 且不明确加权 | 全覆盖每个验证预测目标；token 数加权 NLL、PPL 和计数断言 |
| P1 | 02/11 `data.py` | Bigram 验证包含训练原句；T5 训练同一源输入对应冲突 target，验证又包含相同的样本 | 更换教学样例：固定可学映射且不复用输入输出对；测试重复/冲突 |
| P1 | 12 `data.py` & fixture | 两个 prompt 对应互相冲突的 chosen/rejected，训练验证又复用同一 prompt | 先明确这是 seen-prompt 机制验证；v3 后续扩展为独立问答属性/不相交评估 |
| P1 | 所有 `metrics.json` | 目前没有统一的数据来源/分割信息，无法比较运行批次与重复性 | manifest: 程序版本/任务/data SHA256/参数/分割协议/指标解释 |
| P1 | `utils/v2io.py` | 缺少工件完整性检查及跨任务 checkpoint 识别；默认输出可复用旧目录 | 追加 manifest 与 schema/sha 验证；将 artifact/source/metric 范围完整展示 |
| P1 | `scripts/prepare_data.py` | 通用下载 JSONL 与监督章节的真实任务 schema 不同 | 数据适配器及 card/schema 检查，不允许未标注数据被当 benchmark |
| P2 | `utils/experiment-viewer.js` | 页面导入指标，但没有对不同 run 的差异、置信区间和版本进行明确展示 | 版本/来源/任务可视化、train/valid 对比、error slicing；本地加载不上传 |
| P2 | 17 章 README | 缺少系统性消融记录、难例与复杂度验证 | 每章实验表 + 正反案例 + 论文与 toy 假设对照 |

## v3.0 本次交付（分批，以 CI 为准）

1. 创建与 `v2.0` 隔离的 `v3.0` 分支，保留 v2.0 稳定性。
2. 引入显式 OOV 统计词表，条件概率全空间严格归一化；更新使用说明和测试。
3. 重构 06/10 字符 LM 数据划分与验证计分，使验证字符未进入训练词表，按全部有效 token 加权；保存词表与实际计数。
4. 修复冲突或重复的教学例子，增加数学断言和评估负例。
5. 加入可检查的实验 `manifest.json`：版本、任务、seed、参数、数据源 hash、checkpoint hash 和真实指标；未来可作为跨运行比较的稳定协议。
6. GitHub Actions 为 `v3.0` 增加自动检查；验证通过再考虑 PR。
7. 根 README 提供 v3 快速路径及局限，不把玩具实验称为标准公开 benchmark。

## v3 下一批（不应被本次提交冒充“已完成”）

- **评价协议**：各章真正的 `--mode eval`，从 checkpoint 在新的数据文件上不训练重算指标；split hash + 句子去重。
- **真实数据桥接**：CoLA/CoNLL/SQuAD/OPUS/GLUE 到明确的 toy schema，严格保留数据许可证、官方 split、标签映射。
- **更可信的任务定义**：T5 动态 span corruption（而非固定示例）、BERT 80/10/10 mask 策略可选、DPO 独立 prompt 特征化、RL 多步骤轨迹、RAG 检索和生成错误分解。
- **课程设计**：逐章源码引用定位、可解答的数学测验、浏览器错误案例分析、三层训练难度和耗时预算。
- **技术/安全质量**：manifest/schema 迁移工具、checkpoint 数据上限与不可信文件处理规则、不同平台和 PyTorch 版本的兼容 CI。

## 验收准则

- 常见词、`<unk>` 及上下文未知时 Bigram 的全部输出概率总和为 1；模型恢复后不变。
- 验证段出现训练未见字符时，模型仍能计算有限 loss，且 `val_oov_tokens>0`；checkpoint vocab 不包含原始未见字符。
- 验证 `val_evaluated_tokens == len(validation_slice)-1`；改变块长度不应漏掉尾部。
- 数据例子不能出现相同输入对应两个不同 target 而没有任何额外条件。
- manifest 文件来自真实运行、包含源码版本/数据指纹，工件哈希与文件一致。
- 所有 17 章 v2 train→restore smoke tests 继续通过，新增回归测试必须在 CI 通过。
