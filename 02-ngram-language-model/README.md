# 02 · N-gram 统计语言模型

> 1980s–2000s · 如何用语料统计预测下一个词？

**一句话抓住核心：** 通过前 n-1 个词作为上下文统计条件频数，使用加一平滑避免零概率。

## 运行 (先离线后扩展)

```bash
python 02-ngram-language-model/demo.py 
python -m http.server 8000  # 访问 /02-ngram-language-model/index.html
```

## 从输入到输出的关键步骤

1. **切词**：将句子映射为 token 序列。代码锚点：`sentence.split()`
2. **加入边界符**：句首 <bos>、句末 <eos> 保持序列长度统计。代码锚点：`['<bos>'] + words + ['<eos>']`
3. **统计转移**：统计 context 与 next token 联合出现次数。代码锚点：`counts[context][word] += 1`
4. **平滑预测**：加一平滑使未见词依然有概率。代码锚点：`(count + 1) / (total + V)`

## 核心公式

```text
P(w|h) = (count(h,w)+1) / (count(h)+|V|)
```

## 为什么它比前一代有效？

通过前 n-1 个词作为上下文统计条件频数，使用加一平滑避免零概率。

## 重点阅读代码

打开 [demo.py](./demo.py)，先读数据定义，再读核心函数，最后读 `if __name__ == "__main__"`。关键算法旁有中文注释；不需要先学复杂工程框架。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/Salesforce/wikitext
- 使用建议：WikiText-2 (wikitext-2-raw-v1)，与 nanoGPT 的小语料启动方式相同：先 toy，再大文本。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://aclanthology.org/J07-2002/ (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** 上下文长度固定；稀有 n-gram 稀疏，无法共享近义词统计。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。

## v2.0 · 完整 MVP 入口

本章现已拆分为 `data.py`（训练/验证样本与边界）、`model.py`（核心算法）、`project.py`（任务训练/验证/推理）、`demo.py`（运行命令与 loguru）。输出包括 `metrics.json`、`examples.jsonl` 和 `run.log`，且**不在模块导入时执行训练**。

```bash
pip install -r requirements.txt
python 02-ngram-language-model/demo.py --steps 1 --output-dir outputs/02-ngram-language-model
cat outputs/02-ngram-language-model/metrics.json
```

**独立验证的定义**：从训练语料估计 bigram 条件概率，在 held-out 文本上计算每 token NLL 和困惑度；另执行贪心生成。

**实验建议**：改变输入集、训练步数（统计方法修改数据而非增加无意义 epochs）、种子，并比较工件中每条结果，而不仅看一个标量。
