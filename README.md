# featurebench-codespec2 迁移包（2026-10-24）

FeatureBench full split（200 题）上 5 个方法的**完整运行产物** + 我们的方法源码 +
中间产物 + 依赖清单。用于在另一台服务器上分析 / 复现。

## 目录结构

```
featurebench-codespec2-20261024/
├── Agent/
│   ├── arch_agent_mini_swe/        # ★ 我们的方法（arch-agent-max-full 跑的就是它）
│   └── codespec2/                  # CodeSpec2（后续方法）
├── deep-swe/
│   ├── codespec2_pier/             # pier 适配器（CodeSpec2PierAgent 等）
│   └── run_codespec2*_pier.sh      # 启动脚本
├── featurebench-harness/           # FeatureBench 官方框架（featurebench/ 源码包）
│   ├── featurebench/infer/agents/arch_swe_agent_simple.py   # 我们方法的框架适配器
│   ├── featurebench/infer/agents/codespec2_agent.py         # codespec2 接入
│   ├── pyproject.toml / uv.lock / requirements.txt / config.toml
├── data/
│   ├── full.jsonl                  # FeatureBench full split，200 条（200 个唯一 instance_id）
│   └── full-00000-of-00001.parquet # 同一 split 的 HF parquet
├── trajectories-full/              # ★ 5 个方法 × 200 题的完整 run_outputs（见 INDEX.md）
│   ├── mini-swe-full.tar.gz
│   ├── claude-code.tar.gz.part-00 / .part-01
│   ├── openhands-full.tar.gz
│   ├── arch-agent-max-full.tar.gz.part-00 / .part-01        # 我们的方法（含完整中间产物）
│   ├── rtadev_agent_mini_swe_only-full.tar.gz.part-00 / .part-01
│   ├── outcomes.jsonl              # 995 行，instance_id → resolved/completed/error
│   ├── SHA256SUMS.txt
│   └── INDEX.md
├── intermediates/                  # 补充：CodeSpec2 的 spec / semantic_ir 产物（见 INDEX.md）
│   └── featurebench-codespec2-artifacts.tar.gz   # 8 full + sample 运行
├── env/
│   ├── conda-env-openhands.yaml            # conda 环境 openhands（Py3.12.13，pier 运行环境）
│   ├── pip-freeze-openhands.txt            # 该环境 566 个包
│   ├── pip-freeze-featurebench-venv.txt    # FeatureBench .venv 的 69 个包
│   └── python-version.txt
└── README.md
```

## 5 个方法（full split，各 200 题）

| 方法 | 压缩包 | agent | 轨迹 |
|---|---|---|---|
| mini-swe-agent | `mini-swe-full.tar.gz` | `mini_swe_agent` | `mini_swe_agent_output.log` |
| Claude Code | `claude-code.tar.gz.part-*` | `claude_code` | `claude_code_stream_output.jsonl` |
| OpenHands | `openhands-full.tar.gz` | `openhands` | `trajectory.json` |
| **我们的方法** | `arch-agent-max-full.tar.gz.part-*` | `arch_swe_agent_simple` | `design/test/implement.traj.json` |
| RTA-Dev | `rtadev_agent_mini_swe_only-full.tar.gz.part-*` | `rtadev_agent_mini_swe_only` | `01_product_manager.log` … `04_programmer.log` |

每卷都含完整 `run_outputs/`（`infer.log`、`run.log`、`patch.diff`、agent 输出）、
`eval_outputs/`（每题 `report.json`）、`output.jsonl`、`report.json`。

**我们的方法额外含完整中间产物**（195/200 题）：
`architecture_spec.py`、`f2p_tests.py` / `p2p_tests.py`、`summary.json`、
`static_issues.md`，以及 design / test / implement 三阶段完整轨迹。
详见 `trajectories-full/INDEX.md`。

> 分卷包需要先合并：
> `cat arch-agent-max-full.tar.gz.part-* | tar -xzf -`

## 目标机复现

```bash
# 1) 运行环境（pier / codespec2 / arch_agent_mini_swe 用）
conda env create -f env/conda-env-openhands.yaml
# 或已有环境时：pip install -r env/pip-freeze-openhands.txt

# 2) FeatureBench 框架
cd featurebench-harness && uv sync          # 用 uv.lock 精确还原
# 或 pip install -r ../env/pip-freeze-featurebench-venv.txt
```

包内目录布局与宿主一致，`deep-swe/codespec2_pier/agent.py` 通过
`parents[1]` 定位仓库根，因此 `Agent/` 与 `deep-swe/` 的相对位置不要改。

## 运行

```bash
# A. FeatureBench 框架内跑 full split（200 条）
cd featurebench-harness
cp config.toml config.local.toml    # 按需改 model / docker 配置
fb infer --config-path config.local.toml --agent arch_swe_agent_simple --split full
fb infer --config-path config.local.toml --agent codespec2 --split full

# B. 直接跑单仓库
python Agent/codespec2/run_agent.py \
  --workspace /path/to/repo --task-file /path/to/task.jsonl --model "$MSWEA_MODEL_NAME"

# C. 走 pier（DeepSWE 任务）
cd deep-swe && bash run_codespec2_pier.sh -p tasks/<task> --model deepseek/deepseek-v4-flash
```

## 注意事项

- **未包含凭据**：`deep-swe/.env.local`（API key / base_url）不随包分发；
  `featurebench-harness/config.toml` 中 key 已替换为 `<YOUR_API_KEY>`。
- **轨迹日志已清洗**：原始 `infer.log` / `warmup.log` 里的 `MSWEA_API_KEY` /
  `DEEPSEEK_API_KEY` 已统一替换为 `<REDACTED>`。
- **未包含**：DeepSWE `tasks/`、`subtasks/`、`jobs/`（任务与轨迹）、`runs/` 原始输出、`.git`。
- **离线 runtime**：`codespec2_pier` 需要的 `.codespec2_runtime/site-packages`
  会在首次运行时从 `openhands` 环境自动重建（见 `_ensure_offline_runtime_bundle`），
  前提是该环境装有 `mini-swe-agent` 与 `pyflakes`（pip freeze 已包含）。
- 5 个方法的轨迹卷解压后合计约 3.2 GB，压缩后 464 MB（分卷后单文件 ≤ 90 MB）。
- conda 导出为 `--no-builds`，目标机需同为 linux-64。
