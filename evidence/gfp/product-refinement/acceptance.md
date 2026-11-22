# GFP 用户验收

冻结基准为 `c5941701cd531f58cd810849c999589be344ada7`。本次只运行用户流程和自动测试。正式训练、450 次档案查询、四模型日志、原配对审计和原报告均沿用冻结文件。

| 验收项 | 实际证据 |
|---|---|
| 配置自己的单模型 API | 已有 OpenAI 兼容向导验证原生工具与结果回传。本次两轮 `gfp init / continue` 使用同一真实用户 profile，共接受两个原生评分工具调用和两次最终选择。[向导收据](../configured-api-receipt.json) · [第一轮模型日志](round-01/agent.json) · [第二轮模型日志](round-02/agent.json) |
| 菜单外突变和实测边界 | 首轮载入 96 条公开实测及 2 个菜单种子。两轮均推荐 5 条完整候选序列，各有 4 条菜单外。程序只从当前观测拟合预测器，所有推荐行的 `measured_fitness` 留空。[初始菜单](initial-menu.csv) · [第一轮预测](round-01/recommendations.csv) · [第二轮预测](round-02/recommendations.csv) |
| 导入实测并继续 | 首轮提交后，只有一条推荐在公开档案中有匹配测量。将该测量按用户 CSV 导入后累计 97 条观测，其余首轮候选保留待测。记录保留了首次续跑因响应长度上限失败以及调整用户模式上限后的成功运行。[导入 CSV](published-feedback.csv) · [第二轮状态](round-02/state.json) · [失败记录](attempts/round-02-token-limit/agent.json) |
| 用户 Decision Auditor | 在 API 凭证未设置时核对两轮历史、6 条证据引用、1 条返回实测及其预测残差。记录一次推荐集变化。没有可比反事实，反馈收益为 `NOT EVALUABLE`。[审计报告](audit/audit_report.md) · [结构化轨迹](audit/decision_traces.jsonl) |
| 科研结果精修 | 用冻结日志计算逐轮新测量的最佳值、均值及相对 RF 的配对差，保留高估和低估失败案例。精修 PDF 编译并逐页渲染，共 5 页。[精修报告](../../../output/pdf/gfp-scientific-agent-refined.pdf) · [原正式报告](../../../output/pdf/gfp-scientific-agent.pdf) |
| 自动测试与归档校验 | 完整环境 64 项通过。便携锁定环境 63 项通过，跳过需要可选 Anthropic SDK 的原生传输测试。验证了拒绝无效菜单外替换、前序状态和导入文件校验，以及冻结 Qwen 提案请求仍保持相同字节结构。 |

从克隆、安装和建立 profile 开始见 [README](../../../README.md)。已归档的两轮用户研究可直接执行独立审计，无需数据下载或 API：

```bash
python run.py gfp audit --state evidence/gfp/product-refinement/round-02/state.json --output runs/product-audit
```
