# AGENTS.md — nano-nlp 课程开发规则

## 项目目标
- 像 nanoGPT 一样可读、可改、可运行；不要为了“架构整洁”把一个算法拆成十层封装。
- 每章只讲 **一项最重要的历史突破**，必要时保留简化假设并在 README 明说。
- 从 README → 可视化 index.html → demo.py → tests，形成可以自查的学习闭环。

## 结构约束
- 阶段目录必须采用两位数递增的 01-xxx-xxx 命名。
- 每章必须有 README.md、index.html、demo.py。
- 第三方论文只保留官方页面/PDF URL，不直接复制受限制的全文到仓库。
- 共享稳定数学工具放 utils/，不把所有算法抽象为庞大基类。

## Python 风格
- Python >=3.10；stdlib 优先，神经模型允许依赖 torch。
- 一章默认一个 demo.py；入口放在 if __name__ == "__main__"。
- 代码结构为 Data → Model/Algorithm → Loss/Decode → main；不要隐藏核心数学。
- 核心代码必须有解释“为什么”的中文注释（非每行翻译），尤其注释张量形状、mask、归一化和目标函数。
- 默认 toy 数据、CPU 可运行、随机种子固定，不能依赖 API key 或联网。
- 若算法只是教学近似（例如简化 GRPO、MLM 或 T5），必须在 README 标注不等价之处。
- 实验输出用 "RESULT:" 标记，方便自动化 smoke test。
- 数据加载和外部下载应独立于模型训练过程；不自动执行远程数据集脚本。

## HTML 风格
- 优先可交互的因果链、计算节点、参数微实验；避免截图式架构图。
- 禁止第三方 CDN 依赖，浏览器可以直接打开 index.html。
- 每个可视化步骤对应 Python 搜索锚点；支持鼠标点击与键盘左右方向键。
- 复用 utils/viz.js 和 utils/viz.css；可为单章再增加专有图形。
- 颜色/布局保证小屏可读、对比足够、交互有文字反馈。

## 数据与版权
- 使用 datasets.json 保存确切 Hugging Face repo、config、split、field、license。
- 不把大型训练数据、密钥、缓存或大模型权重提交到 Git。
- 不默认传入不可信远程代码选项；遇到老版 dataset script 需要人工安全导出。
- 每章引用一篇经典论文，保留原始 URL，注明跨版本差异与简化范围。
- 若数据集有 non-commercial 等限制，应直接说明，不默认允许商用。

## PR 完成条件
- python -m unittest discover -s tests -v
- 核心公式和代码符合形状/因果掩码/标签边界约束。
- 无网络默认路径可运行；查看 GitHub Actions 通过后再合并。
- 不能以 toy loss 替代真实性能评估。


## v2.0 — 深度 MVP 准则

参考 [V2_ROADMAP.md](V2_ROADMAP.md)。每章最小验收不是 `demo.py` 打印数值，而是针对真实任务类型实现可检查的 `fit/train → evaluate → predict`；可复用参数模型必须能够保存、加载并独立推断。

- 章节 `model.py` 保留模型核心计算，`data.py` 只负责输入及划分，`project.py` 明确组合训练、验证、推断，`demo.py` 仅负责 CLI；避免通用封装隐藏核心数学。
- 使用 `loguru` 进行结构化实验日志。sink 仅在 `utils/runtime.py` 的 CLI 入口配置，其他模块 `from loguru import logger`；训练数据、每个 epoch、关键指标以及异常都应有可追踪记录。注意隐私、运行成本、避免高频 token 级日志。
- 注释解释**为何这样做、关键张量形状、泄漏风险与实现约束**。不要逐行翻译代码或用注释重复显然的 Python 语句。
- 必须有每章特定交互和机制验证；统计玩具展示必须与真实实验指标区分，可导入并呈现 JSON 指标。
- 测试必须检查核心正确性、泛化评估划分、可复现性、checkpoint 恢复和合理非法输入。两步 smoke 仅证明基本可执行。

## 可复现实验注意事项（v2.0）

- 从项目根目录执行 `demo.py --steps N --output-dir outputs/<stage>`，独立恢复用 `--mode predict --output-dir <同一路径> --input '...'`。
- `model.py` 不在 import 时下载或训练。只有 `project.run` 才能优化参数和生成训练工件。
- PyTorch 序列化只保存可信模型类的 `state_dict`、词表及 primitive 构造参数；恢复时必须 `weights_only=True`，禁止接受任意 pickle 反序列化。
- 训练历史必须来自 `utils.v2io.record_step` 实际运行；可视化不得把示意曲线标记为真实训练记录。
- 外部数据严格依照 [fixtures/README.md](fixtures/README.md) 的 schema；不能把无标签 Hugging Face 原始文本伪装成监督/检索相关性数据。
- 除纯算法不变量外，还需测试两个独立 Python 进程的 `train→predict`，以及真实数据字段的合法/非法输入。

## v3.0 · 禁止评测泄漏与来源丢失

- 构建 vocab、IDF、规则/词典、分布统计时只能使用训练/索引输入，不得利用验证标签或验证专有 token 做训练前的映射。
- 凡报告 perplexity，必须说明 token 分母并确保验证目标全覆盖（包括不足一个 block 的尾部）。
- `<unk>` 必须是输出空间中可计数且可归一化的类别；未见 token 不能凭空增加概率质量。
- 同一完整输入若对应冲突标签，先检查数据定义，不要用更多训练轮数掩盖不可辨识性。
- 每次运行必须输出 `manifest.json`，记录数据 SHA-256 与 artifact hash；v3 恢复预测前应检查 manifest。哈希用于完整性检测而非签名认证。
- toy/seen-prompt/held-out 与官方 benchmark 指标要在 README 和 metrics 中严格区别；未经独立测试，不使用“性能已验证”表述。
