# Fixture data and real dataset contracts

17 个小样例均为原创、离线、可审核的教学数据，不需要下载任何模型或访问 API。它们测试 **数据 schema 与完整代码路径**，并非公开 benchmark。

## 训练真实小样本的统一命令

```bash
python 10-gpt-causal-lm/demo.py --steps 30 --text-file fixtures/10-gpt-causal-lm.txt --output-dir outputs/gpt-fixture
python 10-gpt-causal-lm/demo.py --mode predict --output-dir outputs/gpt-fixture --input "the"
python 13-rag-retrieval/demo.py --text-file fixtures/13-rag-retrieval.jsonl --output-dir outputs/rag-fixture
python 13-rag-retrieval/demo.py --mode predict --output-dir outputs/rag-fixture --input "which predicts masked tokens"
```

## 外部数据 schema

| 章节 | 格式 | 必须字段 | 额外说明 |
|---|---|---|---|
| 01 | JSONL | sentence:str, accepted:bool | ≥4 独立句子；规则固定，评估是 grammar acceptability |
| 02 | TXT 或 text JSONL | 每行一个句子 | 至少5行，每行至少3词 |
| 03 | JSONL | tokens:str[], tags:str[] | 长度一致，训练拟合状态/发射/转移 |
| 04 | JSONL | type=document: id,text; type=query: query,relevant[] | relevant 指向存在的文档 id |
| 05 | TXT 或 text JSONL | 正文词 token | 滑窗产生正样本，语料需较长 |
| 06 | TXT 或 text JSONL | 正文字节/字符 | 验证分片必须长于 block_size |
| 07 | JSONL | source:int[3], target:int[3] | 源 token 1..3，目标 4..6 |
| 08 | JSONL | tokens:int[4], label:int | tokens 0..3，label 0/1 |
| 09 | JSONL | tokens:int[] | 序列固定长度≤16，token 1..5，0 保留给 MASK |
| 10 | TXT 或 text JSONL | 字符文本 | 训练+验证都有长度≥block_size+2 |
| 11 | JSONL | source:int[5], target:int[5] | sentinel 7/8、目标 EOS=9 |
| 12 | JSONL | prompt:int, chosen:int, rejected:int | 单 token 教学任务，ID 0/1，二答案不同 |
| 13 | JSONL | type=document: id,text; type=query: question,evidence[] | evidence 可空表示无证据，文档 id 必须可追踪 |
| 14 | JSONL | operation:add/multiply, a:int, b:int | reward 由算术执行器计算 |
| 15 | JSONL | id,text,source,quality | quality∈[0,1]，文档 id 唯一 |
| 16 | JSONL | x:float[8], y:float[4] | 非生产回归任务，禁止 NaN/Infinity |
| 17 | JSONL | budget:int | 固定合成幂律参数网格，budget>0 |

**注意：** 数据集网页的许可与可下载形式可能会变。根目录的 `datasets.json` 是来源目录；`scripts/prepare_data.py` 只获取原始样本并不自动转成上面所有任务标签。真实下游任务需要人工审核字段、转换为当前 schema、划分训练/验证，不能把没有标签的文本当成有标签的 benchmark。
