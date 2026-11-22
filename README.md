# FLIP GB1 回顾性预测与复算

本仓库可运行的蛋白质功能是 **GB1 官方固定划分上的回顾性预测**。原考核要求的突变设计、多轮虚拟进化、蛋白质四组对照和反馈审计仍未完成。验收状态及证据见[逐项验收表](evidence/gb1-review/acceptance.md)。

## GB1 预测结果

固定 FLIP 快照有 8,733 条记录，训练 2,691，验证 299，最终测试 5,743。预测目标是相对野生型的测定富集分数，野生型参照为 1，分数越高代表测定中的相对富集越高。它综合反映蛋白质折叠及结合。FLIP 的 265 残基表示来自 PDB 5LDE 的嵌合构建体。当前代码核实的是基准内部归一化参照，完整原生 GB1 序列的身份仍未独立核实。[原始测定](https://pmc.ncbi.nlm.nih.gov/articles/PMC4985287/) · [固定 FLIP 数据说明](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/README.md) · [PDB 5LDE](https://www.rcsb.org/structure/5LDE)

| 模型 | Spearman | Pearson | MSE | AP |
|---|---:|---:|---:|---:|
| 加性 Ridge | 0.750 | 0.641 | 1.043 | 0.701 |
| 成对 Ridge | 0.797 | 0.801 | 0.665 | 0.837 |
| Random Forest | 0.757 | 0.674 | 0.916 | 0.726 |

以上是官方四替换测试集上的结果。AP 将实测分数大于 1 的记录视为正例。训练单点实测完整覆盖的四替换测试记录为 0/5,743，直接乘积模型在该测试集上无法估计。所有输出均为汇总指标。[数据与预测协议](gb1-source.md) · [指标与误差](evidence/gb1/) · [GB1 专题 PDF](output/pdf/gb1-retrospective-review.pdf)

## 运行与复算

~~~bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.txt
.venv/bin/python run.py gb1 --output runs/gb1-review
~~~

命令读取固定版本 FLIP 公共数据并校验 ZIP 和 CSV 的 SHA-256。验证集用于选参，测试集用于最终评价。完整的本地套件可用 `.venv/bin/python -m pytest -q` 运行。

由汇总结果重新编译报告

~~~bash
.venv/bin/python gb1_report.py --output runs/gb1-retrospective-review.pdf
~~~

[基准快照与哈希](evidence/gb1-review/base_snapshot.json) · [复算与本地测试收据](evidence/gb1-review/verification.json) · [AI 使用披露](disclosure.md)

## 独立历史研究

[混凝土实测数据回放](historical-materials.md)保留原有四组对照、四种模型适配器、反馈审计、原始日志与混合研究报告。材料研究结果仅用于对应的混凝土任务。
