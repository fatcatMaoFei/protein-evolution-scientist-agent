# GFP Fitness Predictor V2 and Scientific Agent V2

V2 使用独立冻结的同一 GFP 荧光档案。原 450 次正式测量回放、540 次补充回放、四模型对照和 [V1 五页报告](../output/pdf/gfp-scientific-agent-quality.pdf) 保留。[V2 协议](protocol.md) · [V2 数据划分](../evidence/gfp-v2/partition.json) · [V2 五页报告](../output/pdf/gfp-scientific-agent-v2.pdf)

## 使用自己的 API 和 GFP 测量

在仓库根目录安装 `requirements.txt`，按 [项目快速开始](../README.md#从克隆到首份-gfp-结果)配置 Python 3.12。随后使用自己的 OpenAI 兼容接口建立 `default` profile。向导测试原生工具调用。密钥由环境变量或 Git 忽略的本地配置提供。

```bash
python run.py setup
python -m gfp_v2.open_world init --measurements my_gfp.csv --model blend --model-profile default --output runs/v2-round1
python -m gfp_v2.open_world continue --state runs/v2-round1/state.json --measurements new_measurements.csv --model blend --model-profile default --output runs/v2-round2
```

输入实测 CSV 使用 `mutant,fitness`，可加 `sequence` 做交叉校验。序号与项目固定 GFP 参考一致，例如 `D21N:K101R`，野生型写 `WT`。候选文件为可选的 `--candidates` CSV，允许提供 `mutant` 和 `sequence`。V2 开放模式在软件内重建合法完整序列，并给出预测和风险；未导入实测的行标为 `PENDING`。检查 `state.json` 和 `hypothesis_audit.json` 获取延续状态和预先锁定的比较结果。所有分数应使用同一可比荧光尺度。

需要提交参考 FASTA 和自行预览 Top-K 时，使用 `python -m gfp_v2.lab_platform preview --help`。参考 FASTA 必须与固定 GFP 一致。当前任务只校验氨基酸序列。终止密码子检查需要另行提供编码 DNA 和阅读框。

## 复算已归档实验

从仓库根目录运行。未带 API 的重放从实际归档响应恢复工具调用和决定，仍校验状态、预算和测量时序。

```bash
python -m gfp_v2.partition --output runs/v2-partition.json
python -m gfp_v2.diagnose --output runs/v2-diagnosis
python -m gfp_v2.benchmark select --output runs/v2-prediction
python -m gfp_v2.benchmark test --output runs/v2-prediction
python -m gfp_v2.trials --study acquisition --protocol gfp_v2/protocol.md --output runs/v2-acquisition
python -m gfp_v2.trials --study agent --protocol gfp_v2/protocol.md --replay evidence/gfp-v2/agent --output runs/v2-agent
python -m gfp_v2.trials --study audit --protocol gfp_v2/protocol.md --agent-run runs/v2-agent --replay evidence/gfp-v2/audit --output runs/v2-audit
python -m gfp_v2.report --output runs/gfp-scientific-agent-v2.pdf
python -m pytest -q
```

研究结果见 [预测测试](../evidence/gfp-v2/prediction/)、[传统采集](../evidence/gfp-v2/acquisition/)、[Agent 五组对照](../evidence/gfp-v2/agent/) 和 [反馈审计](../evidence/gfp-v2/audit/)。开放模式的档案外候选及导入演示单独归档。 [开放研究记录](../evidence/gfp-v2/open-demo/)

补充按同一参考 GFP 提供 [位点、单多点及结构分析](../evidence/gfp-v2/supplement/landscape.md)、[带真实工具日志的知识图谱](../evidence/gfp-v2/supplement/knowledge_graph/protocol.md) 和 [带 GFP FASTA 验证及批次回传的模拟实验平台](../evidence/gfp-v2/supplement/lab/README.md)。教授原题按运行证据逐项裁决，功能验收与科学增益分别记录。 [验收矩阵](../evidence/gfp-v2/requirement-matrix.md)
