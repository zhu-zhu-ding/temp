# arch_agent_mini_swe

mini-swe-agent based architecture agent for FeatureBench-style tasks.

The agent now uses a goal-driven loop around one central executable object:

```python
spec_goal = design_spec_goal + test_spec_goal + impl_status
```

Artifacts:

- `artifacts/spec_goal.py`: central executable goal script.
- `artifacts/design_spec_goal.py`: design-side executable spec describing what
  functions/modules should change, call relationships, and architecture
  constraints.
- `artifacts/test_spec_goal.py`: test-side executable spec describing how to
  prove the feature through functional reproduction and integration checks.
- `artifacts/impl_status.py`: executable status script reporting which goals
  the current implementation satisfies and where it still fails.
- `artifacts/goal_decision.json`: Goal Agent scheduling decision for the current
  loop.
- `artifacts/goal_report.txt`: latest central spec execution report.

Loop:

1. Run `spec_goal.py`.
2. Goal Agent reviews the task, code state, spec goals, and report. It does not
   write product code or tests. It decides whether the task is complete and
   which agents to call next.
3. Design Agent maintains `design_spec_goal.py`.
4. Test Agent maintains `test_spec_goal.py`.
5. Implement Agent modifies product code and maintains `impl_status.py`.
6. Repeat until Goal Agent marks the task satisfied and `spec_goal.py` passes,
   or the loop budget is exhausted.

The agent intentionally avoids benchmark oracle fields such as `FAIL_TO_PASS`.
It also avoids fixed helper-driven analysis; each mini-swe sub-agent is told its
objective and uses repository inspection, focused commands, and its own
reasoning.

Run with the conda environment that contains `mini-swe-agent`:

```bash
/home/wpd/miniconda3/envs/openhands/bin/python run_agent.py \
  --workspace /path/to/repo \
  --task "natural-language requirement" \
  --model "$MSWEA_MODEL_NAME"
```

Useful options:

```bash
--stop-after-stage goal
--stop-after-stage design
--stop-after-stage test
--stop-after-stage implement
--task-file task.jsonl
--log-root /tmp/arch-mini-swe-logs
```

Set `ARCH_SWE_AGENT_MAX_LOOPS` to control the loop count. The default is `4`.
