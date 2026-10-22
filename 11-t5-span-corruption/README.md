# 11 · T5 Text-to-Text 与 Span Corruption

2020 · **要解决的问题**：如何把预训练目标与翻译、摘要、QA 统一为 text-to-text？

## 一句话理解

Encoder 读入含 sentinel 的损坏文本，Decoder 自回归生成被移除的片段及相应 sentinel。

```text
input: the <extra_id_0> sat ... <extra_id_1> → target: <extra_id_0> cat <extra_id_1> mat <eos>
```

## 代码阅读路线

1. **随机片段移除**：用 sentinel 替换连续片段，而非每 token 随机遮挡。代码锚点：`SOURCE`。
2. **构造目标序列**：目标按 sentinel 顺序重建原始片段。代码锚点：`TARGET`。
3. **教师强制输入**：Decoder 输入是目标右移一位。代码锚点：`DECODER_INPUT`。
4. **交叉注意力**：Decoder 对 Encoder memory 做 cross-attention。代码锚点：`nn.Transformer`。
5. **序列损失**：对所有目标位置计算 token 预测损失。代码锚点：`F.cross_entropy`。

## 运行方式

```bash
pip install -r requirements.txt  # 需要 torch 的章节；可选
python 11-t5-span-corruption/demo.py
python -m http.server 8000
```

在浏览器打开 [index.html](./index.html)，点击每一步查看对应的 Python 代码位置。所有 toy 示例不下载大模型。

## 可选真实数据集

https://huggingface.co/datasets/Salesforce/wikitext

离线样本用于校验计算机制；外部数据可由根目录 `scripts/prepare_data.py` 获取抽样数据。请核对数据许可证与字段。

## 核心论文（外链 PDF）

https://www.jmlr.org/papers/volume21/20-074/20-074.pdf

## 局限与误区

原版 T5 包含 SentencePiece、相对位置偏置和精心设计的语料混合；本章只是捕捉 span corruption 和 Encoder–Decoder 目标。

## 亲自动手

- 修改一个参数（例如 rank、beta、budget、k、steps），预测结果变化再运行验证。
- 列出此算法的监督信号来自哪里：标签、偏好、最终奖励、检索文档还是无监督统计？
- 运行根目录单元测试，检查边界条件。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 11-t5-span-corruption/demo.py --steps 30 --output-dir outputs/11-t5-span-corruption
cat outputs/11-t5-span-corruption/metrics.json
```

**本章实际可验收任务：** span corruption 的 Encoder–Decoder 训练及自回归片段重建；返回 exact match 与 token accuracy。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 以多个 sentinel 表示被删除的文本 span；encoder 编码损坏文本，decoder 按目标右移 teacher forcing 学习连续生成。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → `inference.py` 恢复权重后执行预测 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 11-t5-span-corruption/demo.py --steps 30 --output-dir outputs/11-t5-span-corruption
python 11-t5-span-corruption/demo.py --mode predict --output-dir outputs/11-t5-span-corruption --input '1,7,3,4,8'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** val_exact_match、val_token_accuracy、原始损坏输入/重建目标/实际生成序列。

**最值得做的消融实验：** 逐步生成时，把 Decoder 输入从真实 target 换为自己预测的 token，量化暴露误差累积。

**不要过度解读：** 本章没有 SentencePiece、相对位置编码和 C4 大语料，属于 span objective MVP。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
