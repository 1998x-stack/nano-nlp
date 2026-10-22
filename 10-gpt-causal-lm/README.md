# 10 · nanoGPT 风格的 Decoder-only GPT

**时代：** 2018–2020

**问题：** 为什么 next-token prediction 配合因果注意力，可以训练生成式基础模型？

## 最小模型的核心逻辑

Token + Position Embedding → Causal Attention Block → LM Head；用右移标签训练，生成时只依赖已见 token。

```text
L = −Σ_t log P(x_(t+1) | x_(≤t))
```

## 逐行阅读指南

1. **字符编码**：字符→ID，训练样本 x 与右移目标 y；在代码中搜索 `ids[i:i+T], ids[i+1:i+T+1]`。
2. **位置与词嵌入**：让模型区分同词在不同位置；在代码中搜索 `token(x) + position(arange(T))`。
3. **因果注意力**：下三角可见，防止看到未来标签；在代码中搜索 `mask = triu(1)`。
4. **残差与输出**：Attention + MLP 残差，线性映射 vocab；在代码中搜索 `head(x)`。
5. **优化与生成**：交叉熵训练后，自回归采样下一个 token；在代码中搜索 `F.cross_entropy(logits, y)`。

## 运行

```bash
pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 25
python -m http.server 8000
```

浏览器访问 [index.html](./index.html)，按“上一步 / 下一步”理解张量经过的计算步骤。CPU 可运行，全部输入直接写在脚本中，默认不联网。

## 扩展真实小数据

- 数据集：https://huggingface.co/datasets/karpathy/tiny_shakespeare
- 下载请查根目录 [dataset catalog](../datasets.json) 和 [prepare_data.py](../scripts/prepare_data.py)。
- 先跑通 Toy 数据和损失，再扩展子集、序列长度和 batch；切勿用 toy loss 推断真实泛化表现。

## 经典论文

https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

## 重要边界

本章只实现单层单批极简 GPT，不包含工业级 DDP、FlashAttention、KV cache、BPE 或大规模训练。

## 实验建议

- 查看每个 Tensor 维度，把 batch/seq/hidden/heads 逐一标注。
- 改变 seed、steps 或窗口大小，观察结果和 loss 是否合理。
- 思考本章模型在前一代模型上解决了什么问题，又留下了什么局限。

## v2.0 · 完整 MVP 实验

本章拆分为 `data.py`（明确训练与验证数据）、`model.py`（注意力/门控/目标函数实现）、`project.py`（训练→held-out 验证→推理）、`demo.py`（loguru CLI）。不再以“输出一个张量”作为验收。

```bash
pip install -r requirements.txt
python 10-gpt-causal-lm/demo.py --steps 30 --output-dir outputs/10-gpt-causal-lm
cat outputs/10-gpt-causal-lm/metrics.json
```

**本章实际可验收任务：** 字符 GPT：训练集/验证集分段、causal mask、CE、PPL、无监督自回归生成。

**研究边界：** 数据集刻意很小，验证指标说明管线完整，不证明真实任务泛化。保留原始论文链接，区别本例与论文的完整训练配方。

## v2.0 深度：任务级的可运行研究 MVP

**完整算法链路：** 将文本映射为字符 ID，训练 Decoder-only Transformer 预测下一字符，验证区间严格位于训练文本之后；从 checkpoint 逐 token 自回归生成。

**读代码的顺序：** `data.py` 输入、切分、目标变量 → `model.py` 核心数学计算 → `project.py` 参数拟合/训练、验证和模型保存 → `inference.py` 恢复权重后执行预测 → `demo.py` CLI 包装。所有参数训练必须在当前章节显式可追踪。

### 训练与独立进程恢复

```bash
python 10-gpt-causal-lm/demo.py --steps 30 --output-dir outputs/10-gpt-causal-lm
python 10-gpt-causal-lm/demo.py --mode predict --output-dir outputs/10-gpt-causal-lm --input 't'
python -m unittest discover -s tests -v
```
一个完备实验会生成 `metrics.json`、`examples.jsonl`、`run.log`、`history.jsonl` 与 checkpoint（`checkpoint.pt` 或 `checkpoint.json`）。神经模型只保存权重张量及构造配置，使用 PyTorch weights_only 加载；经典算法保存可审查 JSON 状态。
### 评估与反例

**本章独立验证的核心证据：** val_nll、perplexity、参数量、生成文本与实际每步训练 loss。

**最值得做的消融实验：** 修改序列最后一个 token，之前位置输出应不变；测量无掩码泄漏时此断言如何失败。

**不要过度解读：** 初学者容易把很低的 toy loss 误认为语言理解能力；并非工业 nanoGPT 完整配方。

### 交互可视化与真实实验记录

上方专属交互用于检查算法公式、张量流或模型机制；底部 Evidence Viewer 只能通过导入实际生成的 JSON/JSONL 显示指标、训练 loss 与预测样例。因此图上没有出现数据，不表示实验失败，而表示学习者尚未导入实验工件。
### 进阶练习与验收

1. 用自己的语言解释本章输入、标签或奖励的来源。哪些输出可以独立核验，哪些只能近似测量？
2. 从训练/验证路径出发，列出三处潜在的信息泄漏来源，并指出代码中的防护点。
3. 不重新训练，删除当前 Python 进程，再使用 checkpoint 恢复推理；比较相同输入的输出。
4. 给出本章方法与之前一代模型的性能、假设或可解释性取舍；指出本章明确没有实现的论文内容。
