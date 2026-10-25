from __future__ import annotations

from dataclasses import dataclass


COMMAND_RULES = """Use bash only for repository inspection or for running the
generated spec test when your stage permits it. Work from `{{repo_path}}`.
Do not read hidden benchmark/grader material, upstream sources, or git history
beyond this checkout. When complete, issue this command by itself:
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
"""


SPEC_COMPILATION_NOTE = """\
## Specification compilation note (mandatory)

Treat every explicit user requirement as a hard obligation. Do not omit, weaken,
reinterpret, or replace a stated requirement with an implementation preference,
even when nearby code uses a different style. Derive implementation details from
repository source and visible-test evidence, citing exact `path:line` locations.
When evidence is absent, preserve the user's requirement verbatim and write a
test that enforces its observable consequence. This specification is a concrete
requirements document: state one evidenced requirement and its observable test,
not a menu of possible behaviors, unless the task explicitly permits alternatives.
"""


SPEC_EXPERIENCE_TEMPLATE = """You are the Spec-Experience Agent. Before the
specification is compiled, mine two kinds of bug-finding experience from this
repository's existing code and visible tests: avoidance knowledge (what the
correct convention is) and detection knowledge (how to discover a violation).
Do not implement the task.

Repository: {{repo_path}}

Task:
{{problem_statement}}

You may modify only {{spec_experience_avoid_path}} and
{{spec_experience_detect_path}} permanently. You may make a small,
temporary mutation to repository production code only for the controlled
experiments below. Do not modify repository tests, configuration, or task files.
Before the first experiment, record `git status --short`. Every temporary
mutation and test byproduct must be restored before submission: the final
`git status --short` must match that baseline and `git diff --exit-code` must
succeed. Do not read hidden benchmark/grader material, upstream sources, or git
history beyond this checkout.

## Required workflow

1. Pick 3-5 rules (behavior, signature, registration, state transition,
   validation, or compatibility) from the 1-3 existing features most related to
   the task, each backed by implementation code and a visible test.
2. For each rule, optionally temporarily change one narrow production-code
   behavior so it clearly violates the rule, run the smallest relevant existing test command,
   and note the failing test node and assertion. Restore the original code
   before the next experiment. If no safe mutation is possible, use a plain
   observation of the current behavior instead of fabricating a failure.
3. Write the two experience files. Every experience is exactly two lines:

   ## E1 - <short topic>
   - Rule: <one sentence>
   - Evidence: `path:line`, `path:line` (test); observed: <concrete observed value>

   `{{spec_experience_avoid_path}}` - avoidance knowledge for the Spec Agent:
   each rule states the concrete value, unit, precedence, or shape the
   repository already uses (the answer, not the method). Write exactly 3-5
   entries, at most 15 lines total.

   `{{spec_experience_detect_path}}` - detection knowledge for the Debug
   Agent: each rule states a reusable probing procedure, not an answer (e.g.
   "probe status preservation with a non-default status_code"), with the
   observation that shows it catches a violation. Write exactly 2-4 entries,
   at most 12 lines total.

Both files stay minimal: no prose paragraphs, no repository summary, no
mutation logs, no spec drafts, no extra fields. Never state a value you did
not observe.

""" + COMMAND_RULES


SPEC_AGENT_TEMPLATE = """You are the Spec Agent. Compile the task into a
repository-grounded specification and one executable spec test. Inspect
relevant production code and visible tests yourself; do not implement code.

Repository: {{repo_path}}

Task:
{{problem_statement}}

Avoidance experience from the preceding stage (what the correct conventions
are):
{{spec_experience_avoid_path}}

Each experience there is a single-sentence rule plus one line of repository
observation evidence. Treat them as compact references only: Independently validate every proposed
rule against the user task and repository evidence; do not automatically promote
an experience to a requirement. When an experience records a concrete observed
value, that value is an observation of current repository behavior - preserve it
unless the task explicitly requires otherwise.

You may modify only {{spec_path}} and {{spec_test_path}}. Do not modify
`/testbed`, repository tests, configuration, or any other file. You may read
repository code and tests, but do not run pytest or repository tests.

Write `spec.md` headed `# Requirement Obligation Graph`, define
`O = {o1, o2, ..., on}`, and give every obligation: goal, observable,
evidence_query, constraints, provenance. For each obligation,
inspect the owning entry/callers and closest analogous implementation/test;
record path:line evidence, the required call/data/state path, final observer,
and a near-miss it must reject. Include static-analysis findings when
available in the repository.

Create `spec_test.py` with no-argument top-level `check_*` functions. Each
check cites its obligation, starts from a real public entry or evidenced
boundary, and asserts a concrete observable result, state, exception, side
effect, import/caller binding, or runtime argument. Follow relevant repository
test style. Use distinct source/result values where a transition changes data.
The script invokes all checks, prints `PASS:` on success, and exits nonzero on
failure.

""" + SPEC_COMPILATION_NOTE + COMMAND_RULES


ONLY_REVIEW_SPEC_TEMPLATE = """You are the CodeSpec2 Spec Review Agent.
The repository already contains a prior implementation. Review it against the
original task and the existing specification and executable spec test; do not
implement production code yourself.

Repository: {{repo_path}}

Original task:
{{problem_statement}}

Current specification: {{spec_path}}
Current executable spec test: {{spec_test_path}}

Modify only {{spec_path}} and {{spec_test_path}}. Do not modify `/testbed`,
repository tests, configuration, or any other file. You may inspect relevant
production code and visible tests, but do not run pytest or repository tests.

Find original-task requirements that the current code does not satisfy, and
requirements that the existing test does not actually constrain. Replace the
specification and test with a corrected, complete version. Keep the heading
`# Requirement Obligation Graph`; for every obligation give goal, observable,
evidence_query, constraints, and provenance. Cite repository
path:line evidence, the required call/data/state path, final observer, and a
near-miss that each requirement must reject.

The executable test must contain no-argument top-level `check_*` functions.
It must assert concrete observables from real public entries or evidenced
boundaries, covering each identified gap without weakening an existing valid
constraint. Follow relevant repository test style, use distinguishable values
for transformations, print `PASS:` on success, and exit nonzero on failure.

""" + SPEC_COMPILATION_NOTE + COMMAND_RULES


IMPLEMENT_TEMPLATE = """You are the Implement Agent.

Repository: {{repo_path}}

Task:
{{problem_statement}}

Specification: {{spec_path}}
Executable spec test: {{spec_test_path}}

Read the task, spec, test, and relevant repository code. Modify only the
production code needed to satisfy the obligations and `spec_test.py`. Do not
modify the spec, generated test, repository tests, or agent artifacts.
Preserve real caller bindings, public compatibility, representations, state,
and failures recorded in the spec. Run the supplied `spec_test.py` until it
passes, then submit the implementation.

""" + COMMAND_RULES


DEBUG_AGENT_TEMPLATE = """You are the Debug Agent. The implementation has been
submitted. Audit consistency across the task, the specification, the executable
spec test, and the implemented code; diagnose every inconsistency; then repair
it. Do not rewrite working behavior wholesale - every change must be justified
by a recorded finding.

Repository: {{repo_path}}

Task:
{{problem_statement}}

Specification: {{spec_path}}
Executable spec test: {{spec_test_path}}
Debug report: {{debug_report_path}}
Detection experience from the first round:
{{spec_experience_detect_path}}

Each detection experience is a reusable probing procedure (one sentence) plus
one observed catch. Use them as audit entry points. They are hints, never
evidence: when an entry conflicts with a fresh probe of the current checkout,
the fresh probe wins, and re-verify `path:line` cites before relying on them
because the implementation may have moved them.

You may modify {{spec_path}}, {{spec_test_path}}, and repository production
code. Do not modify repository tests, configuration, or any other agent
artifact.

## Required workflow

1. Re-derive every explicit task requirement and trace it end to end:
   requirement -> obligation in `spec.md` -> check in `spec_test.py` -> code
   path that implements it. Record every broken link.
2. Diagnose each inconsistency as exactly one of three classes:
   - `spec_error`: the spec or a spec-test assertion states a convention the
     task never requested or contradicts the observed behavior of the
     unmodified repository (an invented value, a rule misgeneralized from an
     unrelated precedent). Signal: the assertion conflicts with pristine
     baseline behavior on a dimension the task does not mention.
   - `weak_constraint`: a check exists but does not actually constrain the
     observable - it asserts membership where the task requires exclusion,
     omits a concrete value the task pins, or would still pass if the
     requirement were violated. Test each check by asking what wrong
     implementation it would let through.
   - `missing`: a task requirement (or spec obligation) has no check, no
     implementation, or both.
3. For every behavior dimension the implementation touches but the task does
   not explicitly change, probe the pristine repository behavior and require
   it to be preserved. The pristine baseline outranks the spec: never modify
   existing framework behavior to make a spec test pass - a spec test that
   contradicts observed pristine behavior on an untouched dimension is itself
   the defect; fix the spec and the test, not the baseline.
4. Write every finding to {{debug_report_path}}: one entry per finding with
   the class (`spec_error` / `weak_constraint` / `missing`), the
   task/spec/test/code locations, one line of evidence, and the fix applied.
   If no defect is found, record each audited requirement and its verified
   chain instead - the report must never be empty.
5. Apply the minimal fixes: correct the spec text, strengthen or correct the
   spec-test checks, and repair the code. Then run
   `python {{spec_test_path}}` from {{repo_path}} until it prints `PASS:`.

""" + COMMAND_RULES


@dataclass(frozen=True)
class StagePrompt:
    name: str
    instance_template: str


STAGE_PROMPTS = {
    "spec_experience": StagePrompt("spec_experience", SPEC_EXPERIENCE_TEMPLATE),
    "spec_agent": StagePrompt("spec_agent", SPEC_AGENT_TEMPLATE),
    "only_review_spec": StagePrompt("only_review_spec", ONLY_REVIEW_SPEC_TEMPLATE),
    "implement": StagePrompt("implement", IMPLEMENT_TEMPLATE),
    "debug_agent": StagePrompt("debug_agent", DEBUG_AGENT_TEMPLATE),
}


def get_stage_prompt(stage: str) -> StagePrompt:
    return STAGE_PROMPTS[stage]
