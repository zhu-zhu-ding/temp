# CodeSpec2

```text
spec_experience -> spec_agent -> implement
```

## Only-review mode

`codespec2_only_review` initializes a task checkout with a previous run's
`artifacts/model.patch`, copies its `spec.md` and `spec_test.py`, then runs one
specification review followed by implementation. It uploads neither the old
checkout nor verifier output, hidden tests, or solution files.

From `deep-swe/`:

```bash
bash run_codespec2_only_review_pier.sh \
  -p tasks/kombu-virtual-queue-dead-lettering \
  --model deepseek/deepseek-v4-pro
```

The default source is the requested `2026-08-19__02-37-37` run. A Pier job
configuration may pass `source_run=/absolute/path/to/run` to choose another
source. The new task workspace commit and agent artifacts are captured by the
resulting DeepSWE job.

1. **Spec-Experience Agent** mutates narrow, analogous production behaviors,
   runs the corresponding visible tests, restores the checkout, and writes 3-5
   evidence-backed experiences to `spec_experience.md`.
2. **Spec Agent** compiles the instruction and those experiences into a complete
   Requirement Obligation Graph, `O = {o1, o2, ..., on}`, and `spec_test.py`.
3. **Implement Agent** changes production code until the refined spec test
   passes.

Artifacts:

- `spec.md`: initial then repository-refined obligation graph.
- `spec_test.py`: one executable test combining architecture and behavior.
- `spec_experience.md`: temporary-mutation experiments and reusable rules for
  the task's specification.
- `summary.md`: stage results.

Run:

```bash
cd /home/wpd/workspace/FeatureBench/Agent
/home/wpd/miniconda3/envs/openhands/bin/python3.12 codespec2/run_agent.py \
  --workspace /path/to/repository --task-file /path/to/task.jsonl \
  --model "$MSWEA_MODEL_NAME"
```
