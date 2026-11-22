# 科研决策审计

审计对象为实测混凝土闭环。审计记录、临时决策和测量查阅均使用独立目录。

## 归档决策轨迹

从已完成的模型运行重建了 72 轮，48 轮发生在首次反馈以后。
已测方案中，有 285/360 个与该轮直接 RF 前五重合。
每轮记录假设、证据引用、提议候选、评分、实际选择、真实测量及下一轮行动。
48 次轮间过渡中有 33 次引用了上一轮刚测得的可追溯证据。
轨迹追溯说明证据可用或被引用。反馈自身的选择效应由下面的配对检查测量。
自由文本假设缺少预先规定的可检验判据时，其支持或反驳状态为 NOT EVALUABLE。

主档案的模型贪心重叠及各轮测量在 [决策轨迹](decision_traces.jsonl)。

## 固定检查点的反馈比较

**NOT EVALUABLE**。llm_agent.choose_candidates() got multiple values for keyword argument 'model'

## 输入与出处

运行 ID decision-audit-20261122T010521153052Z。冻结主实验文件的前后哈希一致性已校验。
方法沿用 LLMDictator 的共用检查点、独立证据分支和实测终局思路。

[冻结审计协议](audit_protocol.md) · [摘要与来源哈希](audit_summary.json)
