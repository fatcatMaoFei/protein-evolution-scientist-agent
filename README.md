# GFP Fluorescence Directed Evolution Scientist

GFP 荧光定向进化科研智能体。从公开实测训练预测器，提出突变、重建完整序列并评分，在提交候选后读取档案实测。支持导入自己的荧光测量和独立决策审计。

**核心结果**　正式研究完成三轮、共 450 次档案查询。核心四组中，随机组在一个种子刷新初始最佳，Model-Only、LLM 和 Knowledge-LLM 均持平。独立补充研究完成 540 次查询，Ridge 与 RF 的新测最佳值配对差异区间跨零。

[五页主报告](output/pdf/gfp-scientific-agent-quality.pdf) · [结果表](evidence/gfp/aggregate/method_summary.csv) · [交付验收与复现](evidence/gfp/quality/acceptance.md)

## V2 科研增量

V2 独立冻结了数据与协议，分开评估预测能力和 Agent 的真实采集收益。96 条训练数据下，混合模型相对 Ridge 的独立测试 Spearman 从 0.400 升至 0.460，高于野生型的 P@5 从 0.15 升至 0.40。主要 AP 指标的配对区间跨零。2,048 条下，混合模型的回归误差较低，高荧光尾部仍需改进。

传统采集比较和五组 Agent 比较分别完成 480 与 600 次新档案查询。相对每块表现较好的非 LLM 策略，普通 Agent 的新测最佳值差为 −0.0211，知识组为 +0.0094，未确立稳定增量。开放研究示例可接入自己的 API，生成完整 GFP 序列并继续导入 CSV 实测。档案外设计保持待测。

[V2 五页报告](output/pdf/gfp-scientific-agent-v2.pdf) · [V2 使用与复现](gfp_v2/README.md) · [原题及七项加分的验收矩阵](evidence/gfp-v2/requirement-matrix.md) · [V2 预测诊断](evidence/gfp-v2/predictor-diagnosis.md) · [V2 冻结协议](gfp_v2/protocol.md)

## 从克隆到首份 GFP 结果

使用 `uv` 和 Python 3.12。以下两轮演示无需 API 密钥。首次下载的 GFP 数据会校验 SHA-256，随后使用本地缓存。

~~~bash
git clone https://github.com/fatcatMaoFei/protein-evolution-scientist-agent.git
cd protein-evolution-scientist-agent
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.txt
.venv/bin/python run.py demo --dataset gfp --output runs/gfp-first
~~~

## 接入自己的模型

~~~bash
.venv/bin/python run.py setup
.venv/bin/python run.py demo --dataset gfp --model-profile default --output runs/gfp-first-llm
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

开启时，每个提案记录一条已观测的 GFP 证据和理由。Agent 在提交前锁定一个主检候选及其已观测参照，判据为主检荧光严格超过参照。导入后，程序按原阈值给出支持、反驳或待测，保留原预测和残差，将历史检验结果和最新观测交给下一轮。`archive_status` 只标记是否存在于公开档案，平台仍需显式导入实测。证据来源可在 `agent.json` 查看，判定见 `state.json` 和用户审计。

可用 `--candidate-file` 提交候选 CSV，必需字段为 `mutant`，可选 `sequence`。开启菜单外提案时，此文件提供菜单种子。关闭开关时，候选限定在所提供的范围。`demo`、`evolve` 和正式归档回放继续使用原有闭合候选模式。

保留每轮整个输出目录及其相对位置。用户审计读取最新 `state.json` 及前序记录，交叉核验推荐序列、预测、证据来源与实测导入，把已返回实测与原预测关联，记录前后推荐变化。命令不调用 API。用户轨迹没有受控反事实时，`audit_report.md` 将反馈收益标记为 `NOT EVALUABLE`。[菜单外实跑与用户验收](evidence/gfp/product-refinement/acceptance.md)

## 科学质量补充验证

独立演示的第一轮 `supported` 只表示预登记的较弱比较判据成立。`D21N:K101R:I123V` 实测 3.5620，超过已锁定的荧光阈值 1.3004，但低于野生型 3.7192。这次比较无法验证提议的生化机制，也未显示候选优于野生型。第二轮仍待测，反馈收益为 `NOT EVALUABLE`。[预登记与实测状态](evidence/gfp/quality/hypothesis/status.json) · [用户审计](evidence/gfp/quality/hypothesis/audit/audit_report.md)

原实验各池中有 4、0、3 条实测超过初始最佳。种子 29 的池内最高值仍比起点低 0.0803。均匀抽 15 条命中超起点变体的理论概率分别为 14.79%、0%、11.30%。这类变体从未进入 RF Top-5，24 个 LLM 实验臂中只有 3 个通过探索位看到过，均未选中。终局最佳受起点约束。新批次最佳反映少量测试找到的最亮变体，批均值反映总体推荐质量，高于野生型命中率以固定参照衡量发现频率。[零 API 诊断](evidence/gfp/quality/diagnosis/diagnosis.md)

独立验证上 Ridge 的 Spearman 为 0.430，RF 为 0.342。上十分位 P@5 是 0.200 对 0.133，P@10 则是 0.133 对 0.163。RF 的树间离散度可标记相对高误差样本，±1.96 倍离散度只覆盖 75.3% 的验证值。原正式 RF 和参数继续用于冻结回放。[模型验证与全部指标](evidence/gfp/quality/model_validation/)

独立补充实验事先冻结 12 个配对区块，各方法三轮共 15 次，共 540 次档案查询，API 调用为 0。主指标为新发现的最佳实测值。Ridge 相对 RF 的配对均值差为 +0.0226 log10，描述性 95% 区间 [−0.0219, +0.0719]，12 个配对中 6 胜、3 负、3 平。此预算下没有足够证据判定哪种预测排序能稳定发现更亮的候选。[冻结补充协议](gfp_quality_protocol.md) · [配对结果](evidence/gfp/quality/acquisition/contrasts.csv) · [验收与证据](evidence/gfp/quality/acceptance.md)

~~~bash
.venv/bin/python gfp_diagnose.py --output runs/gfp-diagnosis
.venv/bin/python gfp_model_validation.py --output runs/gfp-model-validation
.venv/bin/python gfp_acquisition_ablation.py --output runs/gfp-acquisition
.venv/bin/python gfp_hypothesis_example.py --model-profile default --output runs/gfp-hypothesis
.venv/bin/python run.py gfp report --quality --run-dir evidence/gfp --output runs/gfp-quality-report.pdf
~~~

## 研究与工程证据

正式研究使用 ProteinGym 1.3 的 `GFP_AEQVI_Sarkisyan_2016`。官方 51,714 条变体全部通过序列重建检查，35,584 条符合 1–4 点替换范围。野生型实测为 3.7192121319。荧光采用档案的 log10 标度。[数据来源与许可](gfp-source.md) · [冻结协议](gfp_protocol.md)

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

## 历史研究

[GB1 回顾性预测](evidence/gb1-review/acceptance.md) 和[混凝土历史研究](historical-materials.md)保留在各自的归档中。
