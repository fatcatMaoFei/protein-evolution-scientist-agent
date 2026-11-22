# 历史研究 · 混凝土实测数据的决策回放

本页归档 UCI 28 天混凝土抗压强度研究，包含四组方法、四模型和反馈审计。回归和决策结果适用于对应的混凝土历史任务。

[历史混合研究 PDF](output/pdf/scientific-agent-assessment.pdf) · [材料主协议](protocol.md) · [双模型验证](protocol-factorial.md) · [四模型验证](protocol-four-model.md) · [Qwen 格式修订](protocol-qwen-structural.md) · [AI 使用披露](disclosure.md)

## 结果

**主实验**。DeepSeek 闭环比较随机、RF 贪心、普通 Agent 与知识 Agent。每组每种子 15 次测量，共 180 次查询。普通 Agent 减 RF 贪心的最终最佳值差为 0、−4.30、0 MPa。知识组与普通组在三个种子上的最终最佳值相同，分别命中 27/45 和 24/45 个高强度配方。RF 贪心命中 30/45，随机命中 4/45。[四组汇总](evidence/concrete/method_summary.csv) · [逐轮结果](evidence/concrete/rounds.csv) · [选择与已测结果](evidence/concrete/selections.csv)

**稳健性验证**。DeepSeek V4.1 Flash、Claude Sonnet 5.5、GPT-6 Luna 与本地 Qwen3-4B 各有知识 OFF、ON 两组，固定种子 17、29、43。两种共享控制复用主实验。12 个模型×种子配对里，知识增强使最终最佳值持平 10 次，下降 2 次，两次下降均来自 GPT-6 Luna。八个 Agent 组的 360 次选择都落在各自 RF 前 24 名。[模型汇总](evidence/across-models/cell_summary.csv) · [知识效应与交互](evidence/across-models/contrasts_by_seed.csv) · [使用量](evidence/across-models/usage.csv)

**探索性消融**。RF 贪心、RF 加树间分歧、RF/Ridge 等权集成贪心、集成加分歧的平均最佳提升依次为 4.33、2.90、2.47、3.06 MPa。四个策略使用相同的历史实测候选池和实验预算。[消融结果](evidence/exploratory/summary.csv)

## 安装和复现

Python 3.12。归档回放按请求与响应摘要重新执行候选校验、实验真值查阅、模型拟合和统计。它不需要模型 API 凭证。历史复跑入口会校验其依赖的公开数据快照。

~~~bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.txt
.venv/bin/python run.py replay-full --output runs/verified
~~~

在搭载 Apple 芯片的 Mac 上实跑全部模型时，安装 requirements-full.txt，通过进程环境设置 CODEFLOW_API_KEY、CODEFLOW_SONNET_API_KEY、CODEFLOW_GPT_API_KEY，然后执行

~~~bash
.venv/bin/python run.py full --output runs/live
~~~

本地 Qwen 使用公开的固定 MLX 权重，首次实跑需要下载。DeepSeek、Sonnet 和 GPT 走 CodeFlow。模型服务可能更新，实跑的候选决策可以变化。归档保留输入、返回模型 ID 和非敏感原始输出，供确定性回放。各阶段可单独运行，完整选项见 run.py --help。

## 数据、工具与隔离

- [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) 为 CC BY 4.0。425 条 28 天记录归并到 416 个唯一配方，重复测量取中位数。每个种子有 48 条初始观测、96 条验证、176 条候选、96 条最终测试。
- llm_agent.py 将观测分析、假设、候选设计、模型评分与批次审查连成工具流程。知识组增加固定材料知识和本组观测统计。两组使用相同 RF 评分工具和结构化结果要求。
- nonbio.py 为各组维护独立的观测记录。整批候选通过来源、唯一性和预算校验后，oracle 释放实测值。算法逐轮用该组新测量重训。最终测试标签只用于离线评估。
- factorial_runner.py 复核主实验来源、特征顺序、初始测量、数据划分及首轮菜单。归档同时保留请求、原生工具调用、返回模型 ID、候选和用量。
- 后续测量系统可通过人工批准的导入接口追加样本 ID、单位、重复数、协议版本和时间戳。记录导入与候选评分和选择独立。

~~~mermaid
flowchart LR
  P["固定候选特征"] --> M["RF 拟合与评分"]
  O["本组已测记录"] --> M
  O --> A["证据摘要 / 可选知识"]
  M --> L["模型提出候选并调用评分工具"]
  A --> L
  L --> C["批次审查"]
  C --> R["测量 oracle"]
  H["候选池隐藏实测标签"] --> R
  R --> O
~~~

## 审阅入口

- [材料来源和哈希](evidence/concrete/source.json) · [配对拆分](evidence/concrete/splits/) · [多模型证据校验](evidence/across-models/validation.json)
- [DeepSeek 日志](evidence/concrete/arms/) · [Sonnet 日志](evidence/sonnet/arms/) · [GPT 日志](evidence/gpt/arms/) · [Qwen 日志](evidence/qwen/arms/)
- [预测失败案例](evidence/concrete/failures.csv) · [候选分布](evidence/across-models/candidate_distribution.csv)
- 运行 .venv/bin/python -m pytest -q 可执行自动测试。运行 .venv/bin/python verify.py 可从锁定材料测量和账本复核主实验。

人类作者需复核任务范围、主要主张、原始出处和 AI 披露，再决定提交内容。

## 科研决策审计

审计器直接读取四模型的冻结轨迹，生成可追溯的实验状态、假设、候选评分、选择与下一轮记录。配对检查为独立实验，冻结 RF 权重和候选菜单，仅改变上一轮实测反馈的可见性。与同菜单的 Model-Only 贪心策略一起评分。

~~~bash
.venv/bin/python run.py audit --output runs/audit-traces
.venv/bin/python run.py audit --output runs/audit-live --paired
.venv/bin/python run.py audit --output runs/audit-verified --replay-audit evidence/decision-audit
~~~

第一条重建轨迹。第二条额外调用 CodeFlow DeepSeek，需要进程环境中的 CODEFLOW_API_KEY。第三条复核已归档的审计配对，无模型 API 调用。审计写入新目录并校验主实验输入哈希。

[审计报告](evidence/decision-audit/audit_report.md) · [检查点协议](audit_protocol.md) · [配对结果](evidence/decision-audit/paired_feedback_results.csv) · [审计摘要](evidence/decision-audit/audit_summary.json)
