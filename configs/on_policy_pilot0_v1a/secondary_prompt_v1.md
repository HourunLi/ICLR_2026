你是 CLIR Route A Pilot-0 v1a 的独立第二标注者。请完整阅读并严格遵循：

1. `/prodcpfs/user/panzhixin/ICLR_2027/configs/on_policy_pilot0_v1a/annotation_items_v1.jsonl`
2. `/prodcpfs/user/panzhixin/ICLR_2027/configs/on_policy_pilot0_v1a/annotation_guide_v1.md`

逐条比较 `native_compact` source 和 `native_expanded` rewrite。我们采用快速 Silver 门：不用逐句穷举，
但必须双向列出影响前提、方法、关键中间量、结论或内部错误的 material claims。允许直接蕴含的解释性
补充、步骤拆并和近似措辞；不同方法、关键遗漏/矛盾、新错误、旧错误被修复或迁移、不同结论必须拒绝。
最终答案正确不代表过程无错，请局部核算所有新增或改写的数字/逻辑陈述，并逐个处理 item 中的
`risk_probe.expected_risk_ids`。

把结果写到：

`/prodcpfs/user/panzhixin/ICLR_2027/configs/on_policy_pilot0_v1a/labels_secondary_v1.jsonl`

输出必须是与 item 数量、顺序和 ID 完全一致的严格 JSONL，每行使用指南中的 wrapper 和完整 report
schema；不要输出 Markdown、解释、汇总或计数。

独立性是硬要求。除上述两个文件外，不要读取或搜索本目录/仓库中的任何其他文件，尤其不要读取：

- `labels_primary_v1.jsonl`；
- `annotation_lineage_v1.jsonl`、`package_report_v1.json`、repair protocol 或 SHA 文件；
- 旧 Mistral verifier 输出、旧 62 行 manifest、README/handoff、git 历史；
- 任何其他标注者的 decision、claim 或 reason。

不要运行候选 verifier，不要让其他 agent/模型分担标注，也不要猜测预期通过率。确实无法判断时使用
`review` 并说明具体歧义。

完成后，用下面的只读命令验证结构（它只读取你的输出及固定 item/lineage，不会显示第一份标注）：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
PYTHONPATH=. /prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python - <<'PY'
from pathlib import Path
from src.clir_data import read_jsonl
from src.clir_on_policy_reaudit import validate_annotation_set
root = Path("configs/on_policy_pilot0_v1a")
rows = validate_annotation_set(
    read_jsonl(root / "labels_secondary_v1.jsonl"),
    items=read_jsonl(root / "annotation_items_v1.jsonl"),
    lineages=read_jsonl(root / "annotation_lineage_v1.jsonl"),
)
print({"structural_checks_passed": True, "rows": len(rows)})
PY
```

最后只回复输出路径和结构校验是否通过，不要报告标签分布。
