# GFP 科学质量验收

以 `c594170` 的正式交付和 `8f95e8f` 的用户产品精修为基线。原 450 次正式档案查询、四模型工具日志、配对审计和两份原报告均保留。本轮新增实验单独归档。

| 验收项 | 验证结果 |
|---|---|
| 自主突变与用户闭环 | 在两轮真实 DeepSeek 调用中，评分工具各核对 6 条带已观测证据的提案。第一轮选出 2 条、第二轮选出 3 条菜单外候选，均未列于公开测量档案。完整序列和预测已保存，此五条候选均待测。[第一轮](hypothesis/round-01/) · [第二轮](hypothesis/round-02/) |
| 前瞻假设与反馈 | 主检方案在揭示前锁定 `D21N:K101R:I123V`、实测参照阈值 1.3004、RF 预测 3.0770。实测 3.5620，残差 +0.4851。原记录 `supported` 只表示通过了 1.3004 这一较弱的预登记荧光比较判据。实测仍低于野生型 3.7192，这次比较无法验证提议的生化机制，也未显示设计优于野生型。第二轮仍待测。用户轨迹没有受控反事实，反馈收益为 `NOT EVALUABLE`。[待测到实测的状态](hypothesis/status.json) · [审计](hypothesis/audit/audit_report.md) · [保留的失败续跑](hypothesis/attempts/round-02-format/) |
| 零 API 原设计诊断 | 450 份实测与固定数据一致。三池超起点候选分别为 4、0、3 个。RF Top-5 不含该类候选，LLM 只有三臂在探索位看到且未选中。批次最佳、批均值和 WT 命中分别报告。[诊断](diagnosis/diagnosis.md) · [逐轮诊断](diagnosis/round_diagnostics.csv) |
| 验证集模型选择 | 验证 360 行。Ridge 的整体排序和 P@5 更高，RF 的 P@10 更高。RF 树间离散度对误差筛选有用，验证经验覆盖不足 95%。原正式 RF 保留。[验证分析](model_validation/) |
| 12 个独立配对区块 | 实跑 540 次新档案查询，无 API。Ridge−RF 新实测最佳的均值差 +0.0226，描述性 95% 区间跨零。区块、候选、三轮、随机过程和指标在运行前由 `5e188fd` 固定。[协议](../../../gfp_quality_protocol.md) · [配对数据](acquisition/paired.csv) · [对比](acquisition/contrasts.csv) |
| 测试、复算与 PDF | 最终验收中完整环境 83 项通过，无 API 首次演示完成两轮。此前基础依赖环境有 82 项通过，1 项因未装可选 Anthropic SDK 跳过。诊断、模型验证数据、独立消融和用户审计此前均已在无 API 下复算。主报告为 5 页，已逐页渲染，第 2 页含原三轮进化轨迹。相对于 `914384f`，1,699 个受保护文件的 SHA-256 全部一致。[主报告](../../../output/pdf/gfp-scientific-agent-quality.pdf) · [最终验收收据](final-delivery-verification.json) |

## 复现

从仓库根目录使用已安装的锁定环境。以下命令均无需模型 API。用户审计和报告生成直接读取已有记录。

```bash
python -m pytest -q
python gfp_diagnose.py --output runs/gfp-diagnosis
python gfp_model_validation.py --output runs/gfp-model-validation
python gfp_acquisition_ablation.py --output runs/gfp-acquisition
python run.py gfp audit --state evidence/gfp/quality/hypothesis/round-02/state.json --output runs/gfp-hypothesis-audit
python run.py gfp report --quality --run-dir evidence/gfp --output runs/gfp-quality-report.pdf
```

新调用可使用 `python gfp_hypothesis_example.py --model-profile default --output runs/gfp-live-hypothesis`。它会生成新的模型响应。`8f95e8f` 已在干净 GitHub 克隆中实际安装依赖、测试兼容 API、完成用户演示并离线审计。用户格式的输入、导入值均来自公开测量。
