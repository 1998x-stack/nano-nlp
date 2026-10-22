# 03 · 隐马尔可夫模型与 Viterbi

> 1990s · 给定观测词序列，如何寻找最可能的隐藏标签？

**一句话抓住核心：** HMM 分离起始概率、状态转移概率和发射概率；Viterbi 动态规划在对数空间寻找最优状态路径。

## 运行 (先离线后扩展)

```bash
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

打开 [demo.py](./demo.py)，先读数据定义，再读核心函数，最后读 `if __name__ == "__main__"`。关键算法旁有中文注释；不需要先学复杂工程框架。

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
