# DeepSWE Trajectory Failure Modes and CodeSpec2 Design Response

## Scope and evidence

The current [v1.1 trial index](https://deepswe.datacurve.ai/artifacts/v1.1/trials.json)
contains 27,558 full DeepSWE trials across 61 model/effort configurations and 24
base model families: 14,489 pass, 12,915 fail, and 154 error. It exposes raw
artifact availability but no `critique` annotations, so outcome rows alone do
not support a causal error taxonomy.

The earlier [v1 trial index](https://deepswe.datacurve.ai/artifacts/v1/trials.json)
contains the site's complete cross-benchmark trajectory-analysis sample: 1,620
annotated trials, comprising 90 DeepSWE and 90 SWE-Bench Pro trials for each of
9 agents. The analysis below uses all 810 annotated DeepSWE trials for error
counts. It also reads one raw failed trajectory per agent through the public
[release artifact map](https://deepswe.datacurve.ai/artifacts/v1/release.json)
to distinguish final verifier labels from process failures.

Counts are descriptive of this fixed sample. `FAIL_API_FAILURE`, broken tests,
ambiguous tasks, and external execution errors are kept separate from agent
reasoning/implementation failures.

### v1.1 coverage of all model families

The current index has no causal critiques, but all 24 base model families are
covered below. Effort configurations are aggregated within each family.

| Model family | Trials | Pass | Fail | Error |
|---|---:|---:|---:|---:|
| claude-fable-5 | 2,260 | 1,458 | 729 | 73 |
| claude-opus-4-8 | 2,260 | 1,134 | 1,097 | 29 |
| claude-opus-5 | 2,256 | 1,550 | 686 | 20 |
| claude-sonnet-4-6 | 452 | 135 | 316 | 1 |
| claude-sonnet-5 | 2,260 | 996 | 1,248 | 16 |
| deepseek-v4-flash | 452 | 241 | 211 | 0 |
| deepseek-v4-pro | 452 | 284 | 168 | 0 |
| gemini-3-1-pro-preview | 452 | 53 | 399 | 0 |
| gemini-3-5-flash | 452 | 163 | 289 | 0 |
| gemini-3-6-flash | 452 | 211 | 241 | 0 |
| gemini-3-7-flash | 1,356 | 834 | 522 | 0 |
| glm-5-2 | 904 | 361 | 541 | 2 |
| gpt-5-4 | 452 | 234 | 218 | 0 |
| gpt-5-5 | 1,808 | 960 | 848 | 0 |
| gpt-5-6-luna | 2,256 | 816 | 1,440 | 0 |
| gpt-5-6-sol | 2,260 | 1,440 | 816 | 4 |
| gpt-5-6-terra | 2,254 | 1,095 | 1,159 | 0 |
| grok-4-5 | 452 | 243 | 209 | 0 |
| grok-4-6 | 1,808 | 1,087 | 716 | 5 |
| kimi-k2-7-code | 452 | 138 | 314 | 0 |
| kimi-k3 | 452 | 309 | 142 | 1 |
| muse-spark-1-1 | 452 | 241 | 211 | 0 |
| muse-spark-1-2 | 452 | 248 | 204 | 0 |
| qwen3-8-max | 452 | 258 | 191 | 3 |

Do not read these fail rates as model-specific error *types*. The per-agent
taxonomy in the next section is limited to the 9 agents for which the public
trajectory-analysis sample supplies critiques.

## Shared error modes

Across the 810 annotated DeepSWE trials:

| Mode | Count | What the trajectories show |
|---|---:|---|
| Legitimate pass | 217 | The patch reaches the requested public surface and the verifier agrees. |
| Missed requirement | 332 | Most of a feature is present, but one explicit default, route, method, return shape, state rule, or error behavior is absent. |
| Integration error | 76 | Logic exists but is wired to the wrong file, route, caller, registration point, context field, or repository API shape. |
| API/provider failure | 60 | The rollout ends because of timeout/context/provider failure; this is not evidence of a completed implementation's quality. |
| Regression | 46 | New behavior breaks existing visible tests, builtin overrides, dialects, disassembly, package structure, or another established path. |
| Syntax/build failure | 20 | Duplicate declarations, wrong signatures, unused values/imports, or nonexistent repository methods prevent behavioral tests from running. |
| Wrong logic | 20 | The surface is present but a discriminating value/state case fails, often because conversion, recursion, or cancellation logic is subtly wrong. |
| Incomplete trial | 11 | Exploration or partial edits consume the run; no coherent verified patch is completed. |
| Unverified assumption | 10 | The patch chooses a path, marker shape, output directory, representation, or API convention without checking repository precedent. |
| Wrong file | 7 | A plausible neighboring module is edited while the actual registered caller remains unchanged. |
| Knowledge gap | 3 | A repository/language mechanism is used incorrectly despite available local evidence. |

The dominant pattern is not inability to write code. It is failure to recover the
repository-specific contract before coding. A local happy path then validates
the same assumption embedded in the patch.

## Per-agent profiles from all annotated DeepSWE trajectories

Each row covers 90 annotated DeepSWE trials.

| Agent | Main annotated modes | Characteristic failure pattern |
|---|---|---|
| claude-haiku-4-5 | missed 55; integration 14; syntax 6; wrong file 5 | Commits early to a plausible representation or module; public attribute shape and actual registered endpoint/caller are often not recovered from tests. |
| claude-opus-4-6 | missed 50; API failure 10; integration 5; logic 2 | Produces broad implementations but omits exact defaults/service entries or returns a repository-incompatible helper shape; long runs are also exposed to provider timeout. |
| claude-opus-4-7 | missed 33; integration 6; logic 4; test mismatch 2 | Stronger overall completion, with residual failures concentrated in precise state precedence, mutability/signature details, and route mounting assumptions. |
| claude-sonnet-4-6 | missed 50; API failure 10; integration 8; regression 3 | Implements the feature family but loses exact action paths or repository lifecycle semantics; locally authored tests may exercise the chosen wrong route. |
| gemini-3-1-pro-preview | missed 38; integration 21; regression 13; syntax 5 | Broad multi-file edits outpace repository API discovery; existing method signatures, buildability, and regression surfaces are common casualties. |
| gemini-3-flash-preview | missed 38; regression 15; integration 14; syntax 8 | Edits very early and verifies too little; wrong signatures, out-of-scope variables, unused declarations, and missing dialect/registration paths survive to submission. |
| gpt-5-4 | missed 28; regression 7; logic 4; unverified assumption 3 | Usually tests extensively, but can optimize to focused/self-authored tests while missing a task dimension or an existing override/compatibility contract. |
| gpt-5-4-mini | API failure 34; missed 25; incomplete 5; integration/logic 4 each | Context/provider limits amplify premature edits and incomplete verification; when completed, exact web/API surfaces and actual boundary shapes remain weak points. |
| gpt-5-5 | missed 15; integration 3; unverified assumption 2; logic/regression 1 each | Most trials pass; remaining failures are subtle real-boundary errors, such as placing continuation data in `message` instead of the repository's `context`, even when targeted tests were added. |

## Raw trajectory audit

One failed `FAIL_MISSED_REQUIREMENT` raw trajectory was inspected for each agent.
Only 3 of 9 explicitly read existing repository tests before the first edit.
Eight later ran at least one test or build command, but local verification still
accepted the wrong public contract. Several commands piped test output through
`head`/`tail`, ran checks in the background, or inspected only a focused package,
which can hide the real exit status or omit regressions. One trajectory ran no
test/build command at all.

Representative process failures:

- A snapshot API returned dictionaries and the agent's own test indexed those
  dictionaries, while repository analogs and the required public observer used
  attribute-bearing objects.
- A web save operation was tested at the route the patch had invented, not the
  exact action route named by the task.
- A recursive delegation continuation put tool output in a plausible field, but
  did not inspect the provider request's existing conversation-context boundary.
- Large Go/TypeScript/Rust changes were submitted after focused or background
  checks without a trustworthy full build/regression result.

These traces explain why “run tests” alone is insufficient. The agent must first
derive the right oracle from repository tests and code, then preserve honest exit
status and regression reach.

## CodeSpec2 response

The redesign keeps the original four-stage mechanism:

`spec_agent -> spec_refine -> implement`

It changes the emphasis:

1. `spec-test` reads repository tests, implementation, callers, exports, and
   boundaries before reading its skill. The skill is only an angle generator.
2. The semantic IR carries separate task, repository-test, and repository-code
   evidence and requires an explicit coverage ledger for exact public surfaces,
   caller edges, representations, states, failures, and regressions.
3. `review` derives rubrics from repository evidence first. Every rubric expands
   into normally 4-12 atomic subrubric questions with provenance, required
   observation, a near miss, and an independent evaluation.
4. Rework resolves supported failing subrubrics through the real caller/boundary.
5. Implementation must read tests/code before its first production edit and must
   finish with executable specs, focused tests, relevant regression tests, an
   honest build/typecheck, and final-diff hygiene.
6. All four stage prompts live in `agent_prompts/` and may be overridden per run
   with `--prompt-dir`, enabling trajectory-driven prompt iteration without code
   changes.
