# 01 · 符号规则与上下文无关文法

> 1950s–1980s · 人工规则如何明确描述句子结构？

**一句话抓住核心：** CFG 将语法拆成可递归组合的非终结符；递归下降分析器遍历产生式并返回成功的结构树。

## 运行 (先离线后扩展)

```bash
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

打开 [demo.py](./demo.py)，先读数据定义，再读核心函数，最后读 `if __name__ == "__main__"`。关键算法旁有中文注释；不需要先学复杂工程框架。

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
