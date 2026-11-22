# GFP Fluorescence Directed Evolution Scientist

面向普通 GFP 的荧光定向进化代理。它从公开荧光实验中训练预测器，提出候选的明确氨基酸替换并重建完整序列，调用评分工具，审核批次，随后才从实测档案揭示所选变体的荧光。附带 CSV 用户工作流和独立反馈审计。

正式实验使用 ProteinGym 1.3 的 `GFP_AEQVI_Sarkisyan_2016`，固定 GFP 参考序列。官方 51,714 条变体全部通过序列重建检查，35,584 条符合 1–4 点替换范围。野生型参照由原始实测档案补充，分数为 3.7192121319。目标使用档案的 log10 荧光分数，值越高表示在同一测定下越亮。[数据来源与许可](gfp-source.md) · [冻结协议](gfp_protocol.md) · [科研报告](output/pdf/gfp-scientific-agent.pdf)

## 从克隆到首份 GFP 结果

使用 Python 3.12。下载的数据字节会按固定 SHA-256 校验，随后使用本地缓存。

~~~bash
git clone https://github.com/fatcatMaoFei/protein-evolution-scientist-agent.git
cd protein-evolution-scientist-agent
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.txt
.venv/bin/python run.py setup
.venv/bin/python run.py demo --dataset gfp --model-profile default --output runs/gfp-first
~~~

配置向导输入 profile、OpenAI 兼容 Chat Completions Base URL 和 Model ID，例如 API 服务商提供的 `https://…/v1`。可指定已设置的环境变量名，或在隐藏输入中输入 API Key。向导会发出真实的原生函数调用请求，并把工具结果发送回模型，验证模型能以 JSON 消费该结果后才保存 profile。服务必须支持命名工具调用、工具结果消息和结构化 JSON。远程接口使用 HTTPS，HTTP 仅允许字面回环地址。本地 profile 和密钥文件在 Git 忽略的 `.gfp-local/` 下，密钥文件权限为 `0600`，实验日志不含授权头或密钥。

也可用命令行选项配置，密钥通过现有环境变量传递：

~~~bash
.venv/bin/python run.py setup --profile my-gfp --base-url "$GFP_BASE_URL" --model-id "$GFP_MODEL_ID" --key-env GFP_API_KEY
.venv/bin/python run.py demo --dataset gfp --model-profile my-gfp --output runs/my-gfp
~~~

已用可配置接口实测完整导入流程。输入是公开档案中 96 条实测值的用户格式 CSV。代理提出第一批候选，等 5 个实际分数导入后重训，输出第二批 5 条预测。演示没有使用私有或新产生的湿实验数据。[用户演示与日志](evidence/gfp/user-api-demo/) · [API 验证收据](evidence/gfp/configured-api-receipt.json)

## 运行模式

所有 `--output` 指到新目录。以下命令使用同一个 GFP 数据及评分约定。

| 场景 | 命令 |
|---|---|
| 无 API 的两轮演示 | `.venv/bin/python run.py demo --dataset gfp --output runs/rf-demo` |
| 独立预测基准 | `.venv/bin/python run.py gfp benchmark --output runs/gfp-prediction` |
| Random 与 Model-Only 闭环 | `.venv/bin/python run.py gfp evolve --baseline-only --output runs/gfp-controls` |
| 用户配置的单模型闭环 | `.venv/bin/python run.py gfp evolve --model-profile my-gfp --output runs/gfp-single` |
| 原生单模型闭环 | `.venv/bin/python run.py gfp evolve --backend sonnet --with-controls --output runs/gfp-sonnet` |
| 审计一个已完成的单模型运行 | `.venv/bin/python run.py gfp audit --run-dir runs/gfp-single --output runs/gfp-single-audit` |
| 无 API 复算已完成的四模型正式记录 | `.venv/bin/python run.py gfp replay --from-run evidence/gfp --output runs/gfp-replay` |

原生适配器仍支持 `deepseek`、`sonnet`、`gpt` 和本地 `qwen`。CodeFlow 原生适配器分别从 `CODEFLOW_API_KEY`、`CODEFLOW_SONNET_API_KEY`、`CODEFLOW_GPT_API_KEY` 读取凭证。Apple Silicon 上加载本地 Qwen 或跑全部四模型，使用包含 MLX 的 `requirements-full.txt`。整套实跑命令：

~~~bash
uv pip sync --python .venv/bin/python --require-hashes requirements-full.txt
.venv/bin/python run.py gfp full --output runs/gfp-live
~~~

该入口先跑四种模型的独立 smoke，再执行正式试验、汇总、配对审计和报告生成。归档复算按原请求校验散列，复用实际模型输出，重新核对序列、批次和真实测量。归档复算不需要 CodeFlow 密钥或 Qwen 权重。

## 导入自己的荧光测量

初始 CSV 和反馈 CSV 均需 `mutant,fitness`，可加 `sequence` 交叉核验。突变使用 `K3A:G4D` 这种冒号分隔的格式，前置字母必须是固定参考在该位置的真实氨基酸。野生型写 `WT`。这里的字母只是格式示意。填写与参考 GFP 测定一致的荧光分数尺度。重复测量记录、越界和序列不一致会报错。

~~~bash
.venv/bin/python run.py gfp init --input my_measured_gfp.csv --model-profile my-gfp --output runs/round-01
.venv/bin/python run.py gfp continue --state runs/round-01/state.json --measurements my_round1_fitness.csv --model-profile my-gfp --output runs/round-02
~~~

输出保留完整候选序列、预测分数及待测状态。尚未测量的候选行中 `measured_fitness` 留空。可用 `--candidate-file` 提交自己的 GFP 候选 CSV，必需字段为 `mutant`，可选 `sequence`，包括尚未收录在测量档案中的合法 GFP 变体。每一轮写新状态，之前的状态和导入值可追溯。

## 研究与工程证据

三轮、三个配对种子共完成 450 次 GFP 实测档案查询。核心四组中，随机组在一个种子略超初始最佳。RF、DeepSeek 普通组和知识组均持平。高于野生型的命中数依次为 4、14、11、15，每组 45 次查询。四模型知识比较中，首轮同菜单的 12 个配对有 9 个改变了 Top-5，终局最佳值均未提高。[核心结果与四模型表](evidence/gfp/aggregate/method_summary.csv) · [首轮配对](evidence/gfp/aggregate/knowledge_selection_overlap.csv)

- [预测与留出指标](evidence/gfp/benchmark/metrics.csv) · [三轮四组日志与推荐 CSV](evidence/gfp/formal/deepseek/) · [四模型与知识对照](evidence/gfp/aggregate/)
- [GFP 决策审计](evidence/gfp/audit/audit_report.md) · [配对实测与选择变化](evidence/gfp/audit/paired_feedback_results.csv) · [AI 使用披露](disclosure.md)
- [逐项验收与验证收据](evidence/gfp/acceptance.md) · [每条推荐的完整序列、预测与实测](evidence/gfp/aggregate/selections.csv)
- 旧 [GB1 回顾性预测](evidence/gb1-review/acceptance.md) 和 [混凝土历史研究](historical-materials.md) 均独立归档。GFP 结论来自 GFP 实测荧光档案。
