# 中间产物（spec / IR / spec_test 等）

只含**文本类**中间产物，不含 `*.traj.json`、`*.log`、`*.pyc`、`test_output*`。
原始 run 目录仍在原机器上，可按需回查。

## 1. `featurebench-codespec2-artifacts.tar.gz`（4.1 MB，963 文件）

来源：

- `runs/codespec2/2026-08-16__22-53-02/`：8 个 full-split 实例
- `runs/codespec2_sample/*/`：sample 运行，每个时间戳 5–8 个实例

路径保留为 `codespec2*/run_outputs/<instance>/attempt-1/codespec2/<task>_<ts>/…`：

| 文件 | 含义 |
|---|---|
| `semantic_ir.md` | 需求语义 IR |
| `behavior_spec.py` / `arch_spec.py` | 行为规范 / 架构规范（可执行） |
| `spec_review_report.md` | 规范评审报告 |
| `static_analysis.md` / `static_issues.md` | 静态分析结果 |
| `summary.md` | 各阶段摘要 |
| `patch.diff` | 最终补丁 |

> 注意：FeatureBench 版的 CodeSpec2 产物命名是 `semantic_ir` / `behavior_spec` / `arch_spec`，
> 不是 DeepSWE 版的 `spec.md` / `spec_test.py`。

> DeepSWE jobs 的 spec 产物不随本包分发（按要求只保留 FeatureBench 产物）。

## 解压

```bash
cd intermediates
mkdir -p featurebench-codespec2-artifacts
tar -xzf featurebench-codespec2-artifacts.tar.gz -C featurebench-codespec2-artifacts
```
