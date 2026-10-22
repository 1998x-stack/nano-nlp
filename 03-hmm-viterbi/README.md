# 03 · 隐马尔可夫模型与 Viterbi

> 1990s · 给定观测词序列，如何寻找最可能的隐藏标签？

**一句话抓住核心：** HMM 分离起始概率、状态转移概率和发射概率；Viterbi 动态规划在对数空间寻找最优状态路径。

## 运行 (先离线后扩展)

```bash
python -m pip install -r requirements-classic.txt
python 03-hmm-viterbi/demo.py 
python -m http.server 8000  # 访问 /03-hmm-viterbi/index.html
```

## 从输入到输出的关键步骤

1. **初始化**：起始状态分数与第一个观测的发射概率相加。代码锚点：`log(start[s]) + log(emit[s][word])`
2. **动态规划**：每个状态只保留最佳前驱和对应分数。代码锚点：`max(dp[t-1][p] + log(trans[p][s]))`
3. **回溯指针**：保存前驱状态而不是枚举指数级所有路径。代码锚点：`backpointers[t][s]`
4. **最优序列**：从最后一个时间步逐步回溯。代码锚点：`path.reverse()`

## 核心公式

```text
δ_t(s)= max_{s'}[δ_{t-1}(s') + log A(s',s)] + log B(s,x_t)
```

## 为什么它比前一代有效？

HMM 分离起始概率、状态转移概率和发射概率；Viterbi 动态规划在对数空间寻找最优状态路径。

## 重点阅读代码

推荐先读 [data.py](./data.py) 的任务样本、[model.py](./model.py) 的核心公式实现，再看 [project.py](./project.py) 的训练/验证/保存流程。最后运行 [demo.py](./demo.py)：它只是 loguru CLI 入口，不是算法本体。

## 数据集 (进阶，可选)

- Hugging Face / URL：https://huggingface.co/datasets/eriktks/conll2003
- 使用建议：CoNLL-2003 的 POS/NER 标签序列可用于替换玩具词性数据；注意该数据集的许可与访问方式。
- 所有实验默认内置 toy 数据，无需注册账号、网络或下载。
- 如使用外部数据，须核对 dataset card 的许可、访问限制与训练/验证集划分。

## 核心论文

- https://ieeexplore.ieee.org/document/18626 (在线论文/出版页；未将第三方 PDF 二进制直接复制入仓库)

## 自己动手

- 修改输入，让算法暴露边界条件；记录中间变量如何改变。
- 对照 [交互图](./index.html) 中的步骤，定位 `demo.py` 实现。
- 思考为什么下一代 NLP 技术要改进本章方法。

**常见误区：** HMM 假设当前状态仅依赖前一状态、观测仅依赖当前状态；真实语言依赖常更复杂。

**验收：** 运行结果包含 `RESULT:`；根目录运行 `python -m unittest discover -s tests -v`。

## v2.0 · 完整 MVP 入口

本章现已拆分为 `data.py`（训练/验证样本与边界）、`model.py`（核心算法）、`project.py`（任务训练/验证/推理）、`demo.py`（运行命令与 loguru）。输出包括 `metrics.json`、`examples.jsonl` 和 `run.log`，且**不在模块导入时执行训练**。

```bash
pip install -r requirements.txt
python 03-hmm-viterbi/demo.py --steps 1 --output-dir outputs/03-hmm-viterbi
cat outputs/03-hmm-viterbi/metrics.json
```

**独立验证的定义**：从已标注训练序列估计起始/转移/发射概率，在独立验证序列上计算 token accuracy 并返回 DP 网格。

**实验建议**：改变输入集、训练步数（统计方法修改数据而非增加无意义 epochs）、种子，并比较工件中每条结果，而不仅看一个标量。

## v2.0 深度：完整任务与代码执行路径

**问题与模型：** 从监督标注数据拟合初始、转移与发射频数，再运行 Viterbi；隐状态序列不可通过每个位置独立贪心获得。

**逐个模块阅读：** `data.py` 提供训练/验证任务，`model.py` 描述算法核心，`project.py` 调用训练或统计拟合、计算验证指标并保存工件，`inference.py`（神经章节）实现权重恢复预测；经典章节的恢复方法直接定义在 `project.py`。`demo.py` 只是统一 CLI，避免隐藏算法。

### 运行、保存、重新打开模型

```bash
python 03-hmm-viterbi/demo.py --steps 30 --output-dir outputs/03-hmm-viterbi
python 03-hmm-viterbi/demo.py --mode predict --output-dir outputs/03-hmm-viterbi --input 'dogs run'
python -m unittest discover -s tests -v
```
训练、验证与推理不必处于同一进程。工件目录中包含：`metrics.json`（带 seed 和配置的评估报告）、`examples.jsonl`（逐样本预测）、`run.log`（loguru 日志）、`history.jsonl`（真实优化步骤；统计规则项目可为空）、`checkpoint.json / checkpoint.pt`（仅保存可恢复统计状态或参数张量）。
### 如何衡量这章是不是真的在工作

**可量化输出：** val_token_accuracy、每时刻 log-score 表、路径长度与输入长度相等。

**关键正确性断言：** 在每个时间步手算状态概率并核对动态规划数组。

**建议的对照实验：** 把训练集中 N→V 的计数改小，比较最优路径是否变化。

**能力边界：** 原型采用一阶状态假设和加性平滑，未实现更复杂的 CRF 全局特征。

### HTML 的使用方式

先操作当前页面的专属计算流程与参数实验，再运行上方命令。在页面下方 **真实训练 / 评估结果** 面板选择输出目录中的三个实验文件。专属组件是数学演示，导入组件则只显示实际运行产生的指标、损失与预测；二者不能混为一谈。

### 面试自测

1. 这个模型在该阶段解决了什么具体瓶颈？它借助了哪些条件独立性、优化目标或结构约束？
2. 指出模型中最关键的 3 行代码，以及输入输出 Tensor/状态的具体含义。
3. 不看代码，写出本章评估指标的计算过程，说明哪些改动会造成数据泄漏。
4. 如何构造一个反例证明本章方法的局限？下一代模型做了什么改进？
