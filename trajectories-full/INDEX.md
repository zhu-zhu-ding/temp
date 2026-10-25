# FeatureBench full split（200 题）完整运行产物 — 5 个方法

每个方法都是 FeatureBench **full split 200 题**上的一次完整运行，保留原始目录结构：

```
<方法>/
├── run_outputs/<instance_id>/attempt-1/
│   ├── infer.log                 # 推理全过程
│   ├── run.log                   # 容器执行日志
│   ├── patch.diff                # 最终补丁
│   └── <agent 自己的轨迹/产物>
├── eval_outputs/<instance_id>/attempt-1/report.json   # 测试结果
├── output.jsonl                  # 实例级结果（200 行）
├── report.json                   # 汇总（resolved_ids 等）
└── run_metadata.json             # 运行参数（部分运行缺）
```

## 各方法

| 压缩包 | agent 字段 | 实例 | 说明 |
|---|---|---|---|
| `mini-swe-full.tar.gz` | `mini_swe_agent` | 200 | 轨迹：`mini_swe_agent_output.log` |
| `claude-code.tar.gz.part-*` | `claude_code` | 200 | 轨迹：`claude_code_stream_output.jsonl` |
| `openhands-full.tar.gz` | `openhands` | 200 | 轨迹：`trajectory.json` |
| `arch-agent-max-full.tar.gz.part-*` | `arch_swe_agent_simple` | 200 | ★ **我们的方法**；轨迹：`design/test/implement.traj.json` + `arch_swe_agent_simple_output.log` |
| `rtadev_agent_mini_swe_only-full.tar.gz.part-*` | `rtadev_agent_mini_swe_only` | 200 | 轨迹：`01_product_manager.log` … `04_programmer.log` + `prd.md` / `task_plan.md` |

`output.jsonl` 每行含 `instance_id` / `agent` / `model` / `resolved` 等字段，
解析好的扁平索引见 `outcomes.jsonl`（995 行 = 5 方法合并）。

## 我们的方法：arch-agent-max-full

- 实现：`Agent/arch_agent_mini_swe/run_agent_simple.py`
  （FeatureBench 适配器 `featurebench/infer/agents/arch_swe_agent_simple.py`）
- 模型：`deepseek/deepseek-v4-pro`；200 题全部有 `patch.diff` 与测试结果
- **完整中间产物**（195/200 题具备）位于
  `run_outputs/<instance>/attempt-1/arch_swe_agent_simple/<task>_<ts>/`：
  - `design.traj.json` / `test.traj.json` / `implement.traj.json` —— 三阶段完整轨迹
  - `design.static.log` —— 设计阶段静态检查
  - `artifacts/architecture_spec.py` —— 架构规范（可执行）
  - `artifacts/f2p_tests.py` / `p2p_tests.py` —— 生成测试
  - `artifacts/summary.json` / `static_issues.md` —— 汇总与静态问题
- 另有 `run_outputs/<instance>/attempt-1/arch_swe_agent/task.json`（任务快照）

## 分卷合并

超过 100 MB 的包切成 90 MB 分卷，按文件名顺序拼接即可：

```bash
mkdir -p arch-agent-max-full && cd arch-agent-max-full
cat ../arch-agent-max-full.tar.gz.part-* | tar -xzf -
```

完整性校验：

```bash
sha256sum -c SHA256SUMS.txt          # 单文件
cat *.part-* | sha256sum             # 分卷按 SHA256SUMS 中对应的 .tar.gz 记录核对
```

> `SHA256SUMS.txt` 对分卷记录的是每个 part 的哈希；合并后的整包哈希请以
> `cat parts | sha256sum` 现场计算为准。

## 凭据清洗

原始日志里含运行环境导出的 API key（`MSWEA_API_KEY` / `DEEPSEEK_API_KEY`），
打包前已统一替换为 `<REDACTED>`；可用 `grep -r 'sk-'` 复核（剩余命中均为
`task-specific`、`risk-required` 这类普通单词片段）。

## 注意

- `eval_outputs` 只含每题 `report.json`（不含完整 pytest stdout 大文件）。
- 原始 run 目录：`FeatureBench/runs/experiment_results/full/<agent>/`。
- 5 个方法的 `*.tar.gz` 解压后合计约 3.2 GB。
