# GFP 原题逐项验收

使用真实公开荧光档案完成测量回放。正式运行的冻结代码提交为 `47d1b7893f714694da537361d400967a7919ac97`。三个配对种子为 17、29、43。每臂三轮，每轮五条。每一个种子内使用 96 条初始观测、120 条验证、384 条可查询档案、120 条最终预测留出。有测量结果的候选在批次提交后才揭示分数。[冻结协议](../../gfp_protocol.md) · [正式代码 SHA-256](formal-code.sha256)

| 验收项 | 实际执行证据 |
|---|---|
| 实测数据、评分方向、野生型和序列 | 从固定 ProteinGym 版本下载并核验 51,714 条变体，对每条完整序列执行重建。35,584 条符合 1–4 点范围，从原始档案单独取野生型实测 3.7192121319。[来源和下载哈希](benchmark/source.json) · [来源说明](../../gfp-source.md) |
| 预测训练与独立评价 | Ridge 和 RF 在 3 个配对种子的验证集和留出集均实际训练评价，计算 Spearman、Pearson、MSE、P@5、R@5 和 AP。[预测结果](benchmark/metrics.csv) |
| LLM 假设、具体突变、完整序列、评分与批判检查 | 72 份通过校验的原生工具决策，记录显式替换、完整候选序列、模型预测、证据引用与批次提交。[一份真实工具调用](formal/deepseek/arms/seed-29/D_knowledge_llm/round-01-agent.json) · [全部轨迹](audit/decision_traces.jsonl) |
| 三轮 GFP 虚拟闭环 | 30 个种子×策略实验臂均完成三轮，共 450 次实测档案查询、零未完成实验臂。每轮重新拟合使用该臂的已知记录。[运行状态](aggregate/status.json) · [逐轮反馈](aggregate/rounds.csv) · [完整推荐](aggregate/selections.csv) |
| 四组核心对照 | Random、Model-Only、DeepSeek LLM、DeepSeek Knowledge-LLM 各完成 45 次查询。记录终局最佳、野生型命中率及每个种子的首次命中查询次数。[四组表](formal/deepseek/method_summary.csv) · [首次命中](formal/deepseek/methods.csv) |
| 四模型与知识增强 | DeepSeek、Sonnet、GPT、本地 Qwen 各有三个种子的 ON/OFF 对照。首轮使用同菜单，12 个配对中 9 个改变选择集。所有知识增量的终局最佳差均为零。[四模型表](aggregate/method_summary.csv) · [同菜单候选变化](aggregate/knowledge_selection_overlap.csv) · [模型与知识交互](aggregate/knowledge_interactions.csv) |
| 数据隔离与真值揭示 | 自动测试覆盖重复、跨臂、越界和非法揭示。正式档案可核对真值进入上下文的顺序。[种子 17 划分](formal/deepseek/splits/seed-17.json) · [第一轮请求](formal/deepseek/arms/seed-29/D_knowledge_llm/round-01-request.json) · [批次结果](formal/deepseek/arms/seed-29/D_knowledge_llm/round-01.json) |
| 决策审计 | 重建 72 轮和 48 次轮间过渡。4 个预设检查点完成真实配对调用，4 处 Top-5 改变，1 处单批最佳提高，平均最佳差 +0.0495。主归档前后哈希一致。[审计报告](audit/audit_report.md) · [配对结果](audit/paired_feedback_results.csv) |
| 第三方 API 与 CSV 导入 | 交互向导验证了真实工具调用和工具结果回传。使用单个用户配置的接口完成 96 条历史实测 → 导入 5 条反馈 → 5 条下一批预测。另对 5 条档案外候选运行预测，实测字段全部留空。[接入收据](configured-api-receipt.json) · [用户输入与结果](user-api-demo/) · [档案外候选](novel-user-demo/) |
| 自动测试、无 API 回放和正式报告 | 复算校验 40 张 smoke 与正式科学表及 2 张配对审计表，审计回放实时生成计数为 0。逐页渲染检查 5 页 PDF。[验证收据](verification.json) · [正式 PDF](../../output/pdf/gfp-scientific-agent.pdf) |

正式结果限于这份档案、三种初始状态和当前预算。CSV 接入用公开实测构造用户格式输入。
