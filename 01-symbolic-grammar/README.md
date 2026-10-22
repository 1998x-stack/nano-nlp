# 01 · 符号规则与上下文无关文法

> 1950s–1980s · 人工规则如何明确描述句子结构？

**一句话抓住核心：** CFG 将语法拆成可递归组合的非终结符；递归下降分析器遍历产生式并返回成功的结构树。

## 运行 (先离线后扩展)

```bash
python -m pip install -r requirements-classic.txt
python 01-symbolic-grammar/demo.py 
python -m http.server 8000  # 访问 /01-symbolic-grammar/index.html
```

## 从输入到输出的关键步骤

1. **词汇规则**：终结符如 the/cat/sees 指向词类 Det/N/V。代码锚点：`GRAMMAR['N']`
2. **递归展开**：从 S 出发尝试 NP 与 VP 两个分支。代码锚点：`parse_symbol('S', tokens, 0)`
3. **边界匹配**：终结符必须与输入 token 精确匹配。代码锚点：`if symbol == tokens[pos]`
4. **得到句法树**：所有输入消耗完毕才接受句子。代码锚点：`end == len(tokens)`

## 核心公式

```text
S → NP VP；NP → Det N；VP → V NP
```

## 为什么它比前一代有效？

CFG 将语法拆成可递归组合的非终结符；递归下降分析器遍历产生式并返回成功的结构树。

## 重点阅读代码

推荐先读 [data.py](./data.py) 的任务样本、[model.py](./model.py) 的核心公式实现，再看 [project.py](./project.py) 的训练/验证/保存流程。最后运行 [demo.py](./demo.py)：它只是 loguru CLI 入口，不是算法本体。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/nyu-mll/glue
- 使用建议：GLUE CoLA 可作为语法可接受性进阶实验；本章离线使用自编的5词句子。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://dl.acm.org/doi/10.1145/365153.365168 (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** 规则不等于语义理解；文法覆盖率和歧义会随语言规模急剧增长。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。

## v2.0 · 完整 MVP 入口

本章现已拆分为 `data.py`（训练/验证样本与边界）、`model.py`（核心算法）、`project.py`（任务训练/验证/推理）、`demo.py`（运行命令与 loguru）。输出包括 `metrics.json`、`examples.jsonl` 和 `run.log`，且**不在模块导入时执行训练**。

```bash
pip install -r requirements.txt
python 01-symbolic-grammar/demo.py --steps 1 --output-dir outputs/01-symbolic-grammar
cat outputs/01-symbolic-grammar/metrics.json
```

**独立验证的定义**：人工标注的未见合法/非法句，模型返回完整语法树及接受与否。

**实验建议**：改变输入集、训练步数（统计方法修改数据而非增加无意义 epochs）、种子，并比较工件中每条结果，而不仅看一个标量。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 用产生式 NP→Det N 与 VP→V NP，递归解析输入并明确列举可能语法树。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 01-symbolic-grammar/demo.py --steps 30 --output-dir outputs/01-symbolic-grammar
python 01-symbolic-grammar/demo.py --mode predict --output-dir outputs/01-symbolic-grammar --input 'the cat sees a dog'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** 文法覆盖率（被接受句的比例）、正确拒绝非法结构、树数量。

**关键正确性断言：** 对于不含词语的句子，预测不接受；对于同一句子但改成 the cat sees a dog，应解析成功。

**建议的对照实验：** 修改文法使 NP→NP PP，观察歧义解析树与指数级候选数量。

**能力边界：** 递归下降不保证左递归终止，示例文法有意避开左递归。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
