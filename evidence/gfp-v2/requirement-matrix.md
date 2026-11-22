# 张强教授 GFP 科研智能体考核逐项验收

按已提供题面与补充清单逐项映射。3b-ii、3b-iii、3b-vi、4b–4d、6b、6d–6e 保留题面编号。其余采用任务描述，当前材料未附可逐字校对的完整原题。`PASS` 表示功能实际执行且有证据。科学收益另用 `NOT EVALUABLE` 或实测比较呈现。原正式协议、实测与所有历史模型输出保持冻结。

## 第一部分 · 六项项目任务与关键子项

| 原始要求 | 实际实现 | 运行证据 | 实验结果 | 验收裁决 | 剩余局限 |
|---|---|---|---|---|---|
| 1a 选择蛋白与优化目标 | 固定一种参考 GFP，目标为更高对数荧光 | [测定来源](../gfp/benchmark/source.json) | WT 3.7192121319，明确量纲 | PASS | 一个 GFP 参考与一类荧光测定 |
| 1b 真实数据及野生型 | ProteinGym 1.3 固定源，参考与变体重建和等位校验 | [数据说明](../../gfp-source.md) · [源校验](../gfp/benchmark/source.json) | 51,714 条目录序列通过重建，35,584 条符合 1–4 点范围 | PASS | 档案回放，无新湿实验 |
| 1c 数据清洗与可重复拆分 | 冻结基因型相关标签以外的抽取规则，保留各角色 ID 与版本 | [拆分](partition.json) · [冻结协议](../../gfp_v2/protocol.md) | 23,528 非 WT 基因型跨角色互斥，剔除历史 9,128 个 ID | PASS | 共享 WT 仅作训练锚点与评价参照 |
| 2a 序列特征及拟合 | Ridge、RF、化学属性、成对替换、分段拟合、混合模型 | [模型诊断](predictor-diagnosis.md) · [开发验证](prediction/validation_summary.csv) | 两种预算均由开发数据选中 blend | PASS | 无预训练嵌入；多点特征采样仍稀疏 |
| 2b 排序和回归评价 | 独立测试 Spearman、Pearson、MSE、MAE、Top-K、WT AP | [独立表](prediction/test_summary.csv) · [配对表](prediction/test_paired.csv) | 96 条下 Spearman 0.460，Ridge 0.400；WT AP 差区间跨零 | PASS | 并未证明高荧光尾部普遍提升 |
| 2c 误差与置信度 | 误差分层、风险相关、验证校准的经验覆盖 | [诊断](predictor-diagnosis.md) · [测试逐条预测](prediction/test_predictions.csv) | blend 90% 型经验区间覆盖 90.9% / 91.3%，低预算区间宽 | PASS | 树间离散不能直接当作个体 95% 概率 |
| 3a 智能体工具架构 | 原生工具打分、序列检查、按自身观测更新 | [真实原生工具记录](agent/arms/2201/D_agent/round-01-model-1.json) · [运行状态](agent/status.json) | 48 个获准 Agent 决策，9 个需要无损别名修复 | PASS | 单个主试 LLM，四模型 V1 实验另存 |
| 3b-i 读取历史实测和失败 | 观测、上轮反馈和假设判定有结构化记录 | [后续请求](agent/arms/2208/D_agent/round-02-request.json) | 可核对每条引用属于当时可见证据 | PASS | 引用不保证推理正确或能改善结果 |
| 3b-ii 分析潜在有益位点 | 位点、替换、组合分层与观测关系查询 | [位点结果](supplement/landscape/selection_sites_adjusted.csv) · [知识查询](supplement/knowledge_graph/followup/summary.json) | K158 被 D/E 选中 10 次，按候选池暴露期望为 5.47 次，7 次高于 WT；工具回传实际测过的位点关联 | PASS | 位点偏好及多位点关联不能推出因果 |
| 3b-iii 自主突变 | 可产生合法替换并重建完整参考长度 | [开放首轮推荐](open-demo/round-01/recommendations.csv) | 5 个档案外设计、预测和完整序列 | PASS | 5 个设计仍待实测 |
| 3b-iv 调用预测工具批量评价 | 共同候选池的全部候选均可评分、在决定前调用原生工具 | [工具预检](preflight/native-smoke-retry/) · [五组状态](agent/status.json) | R1 初始 96 个锁定，可选择五项；各臂逐轮移除已测变体 | PASS | 预测风险为启发式，误差可很大 |
| 3b-v 选择实验优先级 | 按预算验证候选 ID、去重、提交后再揭示 | [提交](agent/arms/2208/D_agent/round-01-commit.json) · [测量](agent/arms/2208/D_agent/round-01-measurement.json) | 三轮每轮五项，隔离反馈 | PASS | 别名修复属于格式恢复，保留原始失败 |
| 3b-vi 解释位置、替换与组合 | 原生决策保留引用、假设、分数与策略；位点知识查找有证据 ID | [修正案例](agent-diagnostics/case-refuted-and-cited.json) · [知识查询](supplement/knowledge_graph/) | 有真实理由和候选身份记录。部分理由与模型预测方向不一致 | PARTIAL | 无逐项反事实界定每句解释的因果作用；机制待检 |
| 3b-vii 导入反馈并更新下一轮 | 仅测到的候选进入当臂 CSV/状态，重新拟合并保留判据 | [第一、二轮](agent-diagnostics/case-refuted-and-cited.json) · [真实导入](open-demo/receipt.json) | 2208/D 两次阈值均被反驳，下轮实际引用第一次测量；用户演示 96→97 | PASS | 引用和选择变化本身不能证明收益 |
| 4a 设置可比较的知识开关 | D/E 同模型、同预测器、同候选空间、同预算，E 多可审计知识 | [试验配置](agent/config.json) · [对照](agent/contrasts.csv) | E−D 最佳新测 +0.0305，区间跨零 | PASS | 只是此知识包的一次受控比较 |
| 4b-1 理化性质 | 疏水性、电荷、大小、极性结构化 | [性质与查询证据](supplement/knowledge_graph/) | 可作为设计查询输入 | PASS | 常量描述为一般性质，环境效应仍待检 |
| 4b-2 保守/激进替换 | 依据可查性质差区分替换 | [知识工具](../../gfp_v2/knowledge_graph.py) · [工具记录](supplement/knowledge_graph/) | 可解释替换为何会进入候选判断 | PASS | 规则不能代替 GFP 实测 |
| 4b-3 合理突变数量 | 仅允许 1–4 点替换，WT 用作测定参照 | [协议](../../gfp_v2/protocol.md) · [校验测试](../../tests/test_gfp_v2_policy.py) | 非法输入拒绝，未污染测量集 | PASS | 无插缺和任意参考序列支持 |
| 4b-4 异常序列 | 标准氨基酸、位置、原 AA、完整序列一致性检查 | [平台补充](supplement/lab/) · [合法性测试](../../tests/test_gfp_v2_policy.py) | 拒绝越界、非法替换与不一致序列 | PASS | 输入为蛋白序列，DNA 终止密码子不适用 |
| 4b-5 防止过多突变 | 构造、评分、平台导入均实施 4 点上限 | [测试](../../tests/test_gfp_v2_policy.py) · [平台补充](supplement/lab/) | 连续流程保留原始校验 | PASS | 上限是本档案和试验规则 |
| 4b-6 有益单点及组合 | 区分单点相对 WT、组合频数与非加性 | [景观分析](supplement/landscape/) · [图谱](supplement/knowledge_graph/) | 可查询单点证据并核对多点实测 | PASS | 不能从组合归因到单点机制 |
| 4c 结构化知识图谱 | 实体、变体关系、测量和来源，可查询、随反馈更新 | [图谱比较](supplement/knowledge_graph/case/summary.json) · [根据回传更新](supplement/knowledge_graph/followup/summary.json) | 真实查询回传新测 `Y39H:S175G`，下一次提案引用该证据和共享 `Y39H`，新假设被反驳 | PASS | 引用是证据可追溯，不是组合因果认定 |
| 4d 解释知识效果 | 原知识 ON/OFF 实测、图谱与简单规则的单上下文补充 | [五组对照](agent/contrasts.csv) · [图谱比较](supplement/knowledge_graph/case/summary.json) | 首次配对两方法均最高 3.7204，各 1 次高于 WT；两批四个推荐相同 | NOT EVALUABLE | 单上下文比较，不能估计稳定额外收益 |
| 5a 三轮虚拟定向进化 | 每臂 96 条初值、96 个候选空间、3×5 逐轮揭示 | [轮次轨迹](agent/rounds.csv) · [可复算明细](agent/selections.csv) | V2 600 次 Agent 组查询，传统预测采集 480 次 | PASS | 真值来自历史档案 |
| 5b 四组参考对照 | V1 四组原样留存，V2 五组含风险采集 | [V1 四组](../gfp/formal/deepseek/method_summary.csv) · [V2 五组](agent/methods.csv) | 各组真实预算相同 | PASS | 闭合试验不评价未测新序列的收益 |
| 5c 固定预算与隔离 | 分区独立、先承诺后揭示、每臂独立拟合 | [试验测试](../../tests/test_gfp_v2_trials.py) · [复算](preflight/trial-replay.json) | 读真值前有提交和请求哈希 | PASS | 模型在训练中见过公开档案的可能性不可度量 |
| 6a 结果对照 | 终局最佳、新测最佳、批均值、WT 命中与配对区间 | [传统采集](acquisition/contrasts.csv) · [Agent 对照](agent/contrasts.csv) | 预测增强及 Agent 尚无稳定最佳新测增量 | PASS | 主要估计只有 8 个独立区块 |
| 6b 重要位点 | 候选曝光、策略选择、逐位点真实表现、突变数分层 | [位点与类型分析](supplement/landscape/) | 推荐频率以候选池曝光对照，另列真实 WT 命中 | PASS | 结构邻近或频繁出现不能证明功能关键 |
| 6c 误差和归因 | 生化条件分层，复算同请求贪婪差异，区分评分与选择 | [预测诊断](predictor-diagnosis.md) · [同请求归因](agent-diagnostics/summary.json) | 普通 Agent 有救回也有漏选；知识组也有漏选 | PASS | 同请求回看不能替代因果反事实 |
| 6d 成功、失败和实测假设案例 | 固定规则从预测和配对档案提取准确、高估、低估、胜负及阈值证据 | [案例数据](supplement/case-examples.json) · [科学闭环](agent-diagnostics/case-refuted-and-cited.json) | `D129G:I167V:M233V` 3.7500→3.7494；有实际正例和负例 | PASS | 展示为可复算极值或示例，非发生率估计 |
| 6e LLM 科学决策价值 | 区分引用、行动变化、真实收益和机制；固定分数反馈对照与重复 | [审计](audit/summary.json) · [阈值核算](agent-diagnostics/hypothesis-accounting.json) | 48 项中 39 项陈述超过模型预测本身。反馈前后所选重叠近似同输入重复 | NOT EVALUABLE | 反馈档案仅两区块；尚无稳健增量或生化机制证据 |

## 第二部分 · 提交物与可运行展示

| 原始要求 | 实际实现 | 运行证据 | 实验结果 | 验收裁决 | 剩余局限 |
|---|---|---|---|---|---|
| 3–5 页 PDF | 五页数据、指标、工作流、三轮曲线、四/五组、案例和结论 | [V2 报告](../../output/pdf/gfp-scientific-agent-v2.pdf) · [验证收据](verification.json) | 主报告在页数范围内并逐页检查 | PASS | 详细位点和图谱放补充材料 |
| 源码及依赖 | V2 独立模块和固定 Python 依赖 | [代码目录](../../gfp_v2/) · [锁文件](../../requirements.txt) | 可在干净克隆安装 | PASS | 专用原生 V1 模型适配依赖可选 |
| README 首用说明 | 无 API 演示、单 API、GFP CSV、继续与审计 | [主 README](../../README.md) · [V2 实操](../../gfp_v2/README.md) | 克隆、实际 API 验证、两轮回放 | PASS | 受所选服务可用性影响 |
| 输入/输出和结果展示 | 分区、配置、CSV、决策、失败日志和报告图表 | [采集结果](acquisition/) · [Agent 结果](agent/) · [诊断](agent-diagnostics/) | 逐条记录预测和实测来源，保留负结果 | PASS | 外部新设计无真值保持待测 |
| 复现与数据/AI 来源 | 无 API 重放，记录使用和模型 ID | [复算收据](preflight/trial-replay.json) · [AI 披露](../../disclosure.md) | 科学表与决策及时序复算 | PASS | 已有模型响应能重放，不能保证新生成逐字一致 |

## 第三部分 · 七项考核重点

| 原始要求 | 实际实现 | 运行证据 | 实验结果 | 验收裁决 | 剩余局限 |
|---|---|---|---|---|---|
| 1 科学问题清晰 | 预测与真实发现两主指标 | [协议](../../gfp_v2/protocol.md) | 分开评价 | PASS | GFP 单任务 |
| 2 数据与建模可靠 | 排除历史数据，开发选择和独立测试 | [划分](partition.json) · [测试](prediction/test_summary.csv) | 两层预算及独立验证 | PASS | 同公开测定 |
| 3 Agent 工具闭环 | 设计、评分、锁定假设、揭示与跟进 | [2208 案例](agent-diagnostics/case-refuted-and-cited.json) | 支持及反驳完整保存 | PASS | 部分回合引用不足 |
| 4 知识增强有效性 | ON/OFF 和图谱对规则补充 | [组间对比](agent/contrasts.csv) | 暂无稳定增益 | NOT EVALUABLE | 小样本和表示修复敏感性 |
| 5 公平对照 | 预算、池、初值一致，比较强非 LLM | [配置](agent/config.json) | 600 次真实揭示 | PASS | 只评价可测池 |
| 6 科学分析和解释 | 机制、预测、排序、实测收益分别描述 | [补充数据](supplement/) · [审计](audit/summary.json) | 正负案例均保留 | PASS | 无因果机制或跨数据集验证 |
| 7 工程与复现 | 干净克隆、API、测试和重放 | [验证](verification.json) | 两轮示例可重放 | PASS | 远程 CI 运行受账户条件限制 |

## 第四部分 · 七项加分

| 原始要求 | 实际实现 | 运行证据 | 实验结果 | 验收裁决 | 剩余局限 |
|---|---|---|---|---|---|
| 1 主动学习或强化学习 | `预测均值 + 0.5 × 风险`，每轮更新 | [采集配置](agent/config.json) · [实测对照](agent/methods.csv) | C 组新测最佳 3.8371，A 3.8248，B 3.8434 | PASS | 无稳定超越纯排序；不引入 RL |
| 2 不确定性估计 | 验证校准区间、独立覆盖、误差相关性和 C/B 消融 | [不确定性结果](predictor-diagnosis.md) · [C 对比](agent/contrasts.csv) | RF spread 风险分辨较强，混合的 90% 区间覆盖接近经验目标 | PASS | 不是个体置信区间；风险加权未证增益 |
| 3 单、双、多点比较 | 按突变数比较分布、误差和选择；匹配单点到组合 | [分类](supplement/landscape/archive_by_mutation_count.csv) · [非加性](supplement/landscape/single_multi_summary.csv) | 1–4 点的高于 WT 比率依次为 21.0%、18.4%、12.1%、6.7%；匹配单点加和的 MAE 随 2–4 点由 0.145 升到 0.492 | PASS | 单点覆盖有限、联合测量保留观测语境 |
| 4 结构或保守位点 | 公开 GFP 结构，对齐研究序列并按真实结构坐标诊断 | [结构来源](supplement/landscape/provenance.json) · [结构比较](supplement/landscape/structure_comparison_means.csv) | 1GFL 映射 227/238 位点；低数据探索性 AP +0.0278，P@5 下降 | PASS | 测试已开封，需另做前瞻结构增量验证 |
| 5 知识图谱 | 结构化图谱实际原生工具查询、反馈更新，对照规则 | [图谱补充](supplement/knowledge_graph/) | 查询输出可追踪到实测，不赋予未揭示标签 | PASS | 仅有局部展示，稳定增量另判不可评估 |
| 6 简易交互 Demo | 无 API 体验、自己的 Key/Base URL/Model、GFP FASTA、候选及续跑 | [使用说明](../../gfp_v2/README.md) · [干净克隆收据](verification.json) · [方案](supplement/lab/) | 真实 API 验证及两轮导入复算 | PASS | 仅支持已固定的 GFP 参考 |
| 7 模拟实验平台 | 批次/样本 ID、状态、导出、核验和回传 | [平台回执](supplement/lab/receipt.json) · [第三轮真实模型](supplement/lab/agent-after-return/) | 5 个无测量候选按规则失败，Fitness 留空；两条真实公开档案值回传，99 条观测拟合并进入第三轮 | PASS | 未连接湿实验设备；模拟值不作科研真值 |

## 第五部分 · 项目额外工作

| 原始要求 | 实际实现 | 运行证据 | 实验结果 | 验收裁决 | 剩余局限 |
|---|---|---|---|---|---|
| 四模型 | 旧实验和全部旧响应原样保留 | [四模型归档](../gfp/formal/) | V2 采用一个代表性模型完成公平比较 | PASS | V2 未重新估计跨模型外推 |
| 第三方 API | 兼容 Base URL、Model ID、Key/环境变量 | [真实干净克隆验证](verification.json) | 原生工具调用和工具结果回传通过 | PASS | 每个用户提供商需通过工具兼容测试 |
| 用户连续研究 | 96→97，新增实测指向真实档案，未试新序列继续待测 | [用户回执](open-demo/receipt.json) · [平台补充](supplement/lab/) | 两轮均产出可复算建议 | PASS | 用户新增的是独立档案观测，未验证那 5 个新设计 |
| Decision Auditor | 固定分数、配对反馈与同输入重复、真实终点 | [审计汇总](audit/summary.json) · [回放](preflight/trial-replay.json) | 16 个决策；最佳新测差的区间跨零 | PASS | 科学增量与因果机制不可由当前证据确认 |
