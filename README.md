# GFP Fluorescence Directed Evolution Scientist

面向普通 GFP 的荧光定向进化代理。它从公开荧光实验中训练预测器，提出候选的明确氨基酸替换并重建完整序列，调用评分工具，审核批次，随后才从实测档案揭示所选变体的荧光。附带 CSV 用户工作流和独立反馈审计。

正式实验使用 ProteinGym 1.3 的 `GFP_AEQVI_Sarkisyan_2016`，固定 GFP 参考序列。官方 51,714 条变体全部通过序列重建检查，35,584 条符合 1–4 点替换范围。野生型参照由原始实测档案补充，分数为 3.7192121319。目标使用档案的 log10 荧光分数，值越高表示在同一测定下越亮。[数据来源与许可](gfp-source.md) · [冻结协议](gfp_protocol.md) · [精修科研报告](output/pdf/gfp-scientific-agent-refined.pdf)

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
| 审计自己的导入研究 | `.venv/bin/python run.py gfp audit --state runs/round-02/state.json --output runs/user-audit` |
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
.venv/bin/python run.py gfp init --input my_measured_gfp.csv --model-profile default --allow-novel --output runs/round-01
.venv/bin/python run.py gfp continue --state runs/round-01/state.json --measurements my_round1_fitness.csv --model-profile default --allow-novel --output runs/round-02
.venv/bin/python run.py gfp audit --state runs/round-02/state.json --output runs/user-audit
~~~

`--allow-novel` 是可选开关。开启时，LLM 可提出菜单外的 1–4 点 GFP 替换，程序重建完整序列，并用当前实测数据拟合的 RF 评分。每轮评分清单至少有一条菜单外提案，最终推荐按模型决策产生。入选的菜单外候选在 `recommendations.csv` 中标记为 `origin=llm_off_menu`，所有新推荐均为 `prediction_only`，`measured_fitness` 留空。进入下一轮时，可导入已返回的部分实测，其余推荐继续待测。菜单外表示未出现在当轮菜单中，可以通过实测档案单独核对其是否已有公开测量。

可用 `--candidate-file` 提交候选 CSV，必需字段为 `mutant`，可选 `sequence`。开启菜单外提案时，此文件提供菜单种子。关闭开关时，候选限定在所提供的范围。`demo`、`evolve` 和正式归档回放继续使用原有闭合候选模式。

保留每轮整个输出目录及其相对位置。用户审计读取最新 `state.json` 及前序记录，交叉核验推荐序列、预测、证据来源与实测导入，把已返回实测与原预测关联，记录前后推荐变化。命令不调用 API。用户轨迹没有受控反事实时，`audit_report.md` 将反馈收益标记为 `NOT EVALUABLE`。[菜单外实跑与用户验收](evidence/gfp/product-refinement/acceptance.md)

## 研究与工程证据

三轮、三个配对种子共完成 450 次 GFP 实测档案查询。种子 17、29、43 的初始最佳为 3.8670、4.0292、3.9015。核心四组的 36 个批次中，只有随机组在种子 17 的第二轮超过初始最佳，实测 3.8817。其余三组的终局最佳均持平。高起点和每臂 15 次查询令终局最佳指标在这次实验中区分力有限。[终局与四模型结果](evidence/gfp/aggregate/method_summary.csv) · [逐轮实测](evidence/gfp/formal/deepseek/rounds.csv)

下表每个轮次显示新测量的批内最佳 / 批内均值，再按三个种子取均值。最后一列按种子和轮次减去 Model-Only 再取均值，单位为 log10 荧光。

| 策略 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 相对 RF 最佳 / 均值 |
|---|---:|---:|---:|---:|
| Random | 3.689 / 3.116 | 3.689 / 2.952 | 3.578 / 2.957 | −0.1156 / −0.5364 |
| Model-Only | 3.760 / 3.472 | 3.764 / 3.507 | 3.780 / 3.655 | 0 / 0 |
| DeepSeek LLM | 3.735 / 3.469 | 3.731 / 3.492 | 3.716 / 3.587 | −0.0407 / −0.0289 |
| DeepSeek Knowledge-LLM | 3.812 / 3.300 | 3.841 / 3.672 | 3.758 / 3.519 | +0.0356 / −0.0477 |

知识组的批内最佳均值高于 RF，五条候选的批内均值低于 RF。四模型首轮同菜单的 12 个知识配对中，9 个改变了 Top-5，终局最佳均未提高。预测误差也能影响选样，`E213K` 的预测为 3.7141、实测为 1.3015，而 `K158Q:I161V:K166Q:E235G` 的预测为 2.5268、实测为 3.9489。这些结果是冻结回放上的描述性比较。[知识配对](evidence/gfp/aggregate/knowledge_selection_overlap.csv) · [预测失败记录](evidence/gfp/aggregate/failure_examples.csv) · [精修报告](output/pdf/gfp-scientific-agent-refined.pdf) · [原正式报告](output/pdf/gfp-scientific-agent.pdf)

- [预测与留出指标](evidence/gfp/benchmark/metrics.csv) · [三轮四组日志与推荐 CSV](evidence/gfp/formal/deepseek/) · [四模型与知识对照](evidence/gfp/aggregate/)
- [GFP 决策审计](evidence/gfp/audit/audit_report.md) · [配对实测与选择变化](evidence/gfp/audit/paired_feedback_results.csv) · [AI 使用披露](disclosure.md)
- [逐项验收与验证收据](evidence/gfp/acceptance.md) · [每条推荐的完整序列、预测与实测](evidence/gfp/aggregate/selections.csv)
- 旧 [GB1 回顾性预测](evidence/gb1-review/acceptance.md) 和 [混凝土历史研究](historical-materials.md) 均独立归档。GFP 结论来自 GFP 实测荧光档案。
