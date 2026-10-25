# CodeSpec2 for DeepSWE / Pier

This adapter runs the local CodeSpec2 workflow in a regular Pier task
container:

```text
static analysis → requirement compile → test spec → spec review → spec rework → implementation
```

It uploads only `FeatureBench/Agent/codespec2` and
`FeatureBench/Agent/arch_agent_mini_swe` into `/installed-agent`.  The task
agent receives the normal `/app` checkout and task instruction.  It does not
receive DeepSWE's `tests/`, `solution/`, task metadata, host task directory, or
the parent FeatureBench `Agent` directory.

The derived image build and runtime setup are network-free.  On first use the
adapter copies the existing OpenHands Python 3.12 mini-swe-agent runtime into a
gitignored local cache (`.codespec2_runtime/`); each task then receives it via
Pier's file-upload channel.  This avoids GitHub/PyPI/proxy installation hangs.
Set `CODESPEC2_RUNTIME_SITE_PACKAGES` only when the local source runtime is in
a different Python 3.12 environment.

Run from the `deep-swe` directory:

```bash
bash run_codespec2_pier.sh \
  -p tasks/gql-incremental-graphql-delivery \
  --model deepseek/deepseek-v4-pro
```

Credentials and provider URLs are loaded automatically from the gitignored
`deep-swe/.env.local` file.  Shell environment variables and explicit
`--agent-env` settings override that file when a temporary credential or
gateway is needed.

For an OpenAI-compatible gateway, use the provider spelling matching that
gateway and pass its credentials explicitly:

```bash
bash run_codespec2_pier.sh \
  -p tasks/gql-incremental-graphql-delivery \
  --model openai/deepseek-v4-pro
```

Optional CodeSpec2 limits use Pier's `--agent-kwarg`:

```bash
--ak requirement_compile_step_limit=160 --ak test_spec_step_limit=160 \
--ak spec_review_step_limit=100 --ak implementation_step_limit=260
```

The adapter commits any `/app` change before Pier runs the verifier.  Stage
logs and generated IR/spec artifacts are saved under the job's `agent/codespec2`
directory.  A stage failure is recorded there but does not suppress evaluation
of a partial implementation.

## QuadSpec analysis agent

`QuadSpecPierAgent` launches four independent Mini-SWE agents concurrently.
They write `spec_1.md` through `spec_4.md`; only after all four have finished,
a fifth Mini-SWE agent writes `ambiguity_review.md`. No agent writes tests or
production code. All five documents and trajectories are retained in the job
artifact directory. It is intended to study requirement understanding, not to
produce a scored implementation.

```bash
bash run_quad_spec_pier.sh \
  -p tasks/gql-incremental-graphql-delivery \
  --model deepseek/deepseek-v4-pro
```
