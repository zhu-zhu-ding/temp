from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from codespec2.artifacts import make_artifacts
from codespec2.only_review_agent import CodeSpec2OnlyReviewAgent, OnlyReviewConfig
from codespec2.orchestrator import CodeSpec2Agent, RunConfig, StageResult
from codespec2.prompts import (
    DEBUG_AGENT_TEMPLATE,
    IMPLEMENT_TEMPLATE,
    ONLY_REVIEW_SPEC_TEMPLATE,
    SPEC_AGENT_TEMPLATE,
    SPEC_EXPERIENCE_TEMPLATE,
    get_stage_prompt,
)
from codespec2.replay_agent import CodeSpec2ImplementationReplayAgent, ReplayConfig


class PipelineContractTests(unittest.TestCase):
    def test_spec_prompt_compiles_repository_grounded_spec_and_test(self) -> None:
        for marker in ("{{problem_statement}}", "{{spec_path}}", "{{spec_test_path}}", "Requirement Obligation Graph", "near-miss", "path:line", "visible tests"):
            self.assertIn(marker, SPEC_AGENT_TEMPLATE)
        self.assertNotIn("uncertainty", SPEC_AGENT_TEMPLATE.lower())
        self.assertNotIn("uncertainty", ONLY_REVIEW_SPEC_TEMPLATE.lower())
        self.assertIn("Treat every explicit user requirement as a hard obligation", SPEC_AGENT_TEMPLATE)

    def test_spec_experience_prompt_runs_reversible_evidence_experiments(self) -> None:
        for marker in (
            "{{spec_experience_avoid_path}}",
            "{{spec_experience_detect_path}}",
            "temporarily change",
            "smallest relevant existing test command",
            "git diff --exit-code",
            "git status --short",
            "avoidance knowledge for the Spec Agent",
            "detection knowledge for the Debug",
            "Rule: <one sentence>",
            "observed: <concrete observed value>",
            "at most 15 lines total",
            "at most 12 lines total",
            "Never state a value you did\nnot observe",
        ):
            self.assertIn(marker, SPEC_EXPERIENCE_TEMPLATE)
        self.assertNotIn("Task mapping", SPEC_EXPERIENCE_TEMPLATE)
        self.assertNotIn("Possible spec consequence", SPEC_EXPERIENCE_TEMPLATE)
        self.assertNotIn("{{spec_experience_path}}", SPEC_EXPERIENCE_TEMPLATE)
        self.assertIn("do not automatically promote", SPEC_AGENT_TEMPLATE)
        self.assertIn("Independently validate every proposed", SPEC_AGENT_TEMPLATE)
        self.assertIn("single-sentence rule plus one line", SPEC_AGENT_TEMPLATE)

    def test_standard_pipeline_and_only_review_prompts_are_registered(self) -> None:
        self.assertEqual(
            set(__import__("codespec2.prompts", fromlist=["STAGE_PROMPTS"]).STAGE_PROMPTS),
            {"spec_experience", "spec_agent", "only_review_spec", "implement", "debug_agent"},
        )
        self.assertIs(get_stage_prompt("spec_experience").instance_template, SPEC_EXPERIENCE_TEMPLATE)
        self.assertIs(get_stage_prompt("spec_agent").instance_template, SPEC_AGENT_TEMPLATE)
        self.assertIs(get_stage_prompt("only_review_spec").instance_template, ONLY_REVIEW_SPEC_TEMPLATE)
        self.assertIs(get_stage_prompt("implement").instance_template, IMPLEMENT_TEMPLATE)
        self.assertIs(get_stage_prompt("debug_agent").instance_template, DEBUG_AGENT_TEMPLATE)

    def test_debug_agent_prompt_audits_consistency_and_repairs(self) -> None:
        for marker in (
            "{{spec_path}}",
            "{{spec_test_path}}",
            "{{debug_report_path}}",
            "{{spec_experience_detect_path}}",
            "hints, never\nevidence",
            "spec_error",
            "weak_constraint",
            "missing",
            "pristine baseline outranks the spec",
            "never modify\n   existing framework behavior to make a spec test pass",
            "would still pass if the\n     requirement were violated",
            "the report must never be empty",
        ):
            self.assertIn(marker, DEBUG_AGENT_TEMPLATE)

    def test_all_pipeline_stages_use_max_reasoning_effort(self) -> None:
        with TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.model_config = {"model_kwargs": {}}
            for stage in ("spec_experience", "spec_agent", "implement", "debug_agent"):
                self.assertEqual(agent._model_config_for_stage(stage)["model_kwargs"]["reasoning_effort"], "max")

    def _agent(self, root: Path) -> CodeSpec2Agent:
        agent = object.__new__(CodeSpec2Agent)
        agent.config = RunConfig(repo_path=root, task="task", log_root=root / "logs", model_name="test-model")
        agent.log_dir = root / "logs" / "test"
        agent.log_dir.mkdir(parents=True)
        agent.artifacts = make_artifacts(agent.log_dir)
        agent._stage_counts = {}
        agent._initialize_artifacts()
        return agent

    def test_default_pipeline_runs_experience_then_spec_then_implement(self) -> None:
        with TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            stage = Mock(side_effect=lambda name, _limit: StageResult(name, "Submitted", "", agent.log_dir / f"{name}.traj"))
            agent._run_stage = stage
            agent._should_stop = Mock(return_value=False)
            self.assertTrue(agent.run().passed)
            self.assertEqual(
                [call.args[0] for call in stage.call_args_list],
                ["spec_experience", "spec_agent", "implement", "debug_agent"],
            )

    def test_artifacts_include_spec_experience_and_spec_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            artifacts = make_artifacts(Path(directory) / "run")
            self.assertEqual(artifacts.spec_experience_avoid.name, "spec_experience_avoid.md")
            self.assertEqual(artifacts.spec_experience_detect.name, "spec_experience_detect.md")
            self.assertEqual({p.name for p in artifacts.root.iterdir()}, set())

    def test_implementation_executes_generated_spec_test(self) -> None:
        with TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.artifacts.spec_test.write_text("print('PASS: v1')\n", encoding="utf-8")
            completed = Mock(returncode=0, stdout="PASS\n")
            with patch("codespec2.orchestrator.subprocess.run", return_value=completed) as run:
                self.assertEqual(agent._validate_spec_test(), [])
            self.assertEqual(run.call_args.args[0][-1], str(agent.artifacts.spec_test))

    def test_replay_uses_fixed_spec_and_test_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            (reference / "spec.md").write_text("# Requirement Obligation Graph\n", encoding="utf-8")
            (reference / "spec_test.py").write_text("print('PASS: fixed')\n", encoding="utf-8")
            agent = object.__new__(CodeSpec2ImplementationReplayAgent)
            agent.config = ReplayConfig(repo_path=root, task="task", log_root=root / "logs", reference_artifacts=reference, model_name="test")
            agent.log_dir = root / "logs" / "replay"
            agent.log_dir.mkdir(parents=True)
            agent.artifacts = make_artifacts(agent.log_dir)
            agent._copy_reference_artifacts()
            self.assertEqual(agent.artifacts.spec.read_text(encoding="utf-8"), "# Requirement Obligation Graph\n")

    def test_only_review_initializes_from_patch_and_prior_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess = __import__("subprocess")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "module.py").write_text("value = 'base'\n", encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "base"], cwd=repo, check=True)
            patch = root / "initial.patch"
            patch.write_text(
                "diff --git a/module.py b/module.py\nindex 5d2c82c..fe90277 100644\n--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-value = 'base'\n+value = 'prior'\n",
                encoding="utf-8",
            )
            reference = root / "reference"
            reference.mkdir()
            (reference / "spec.md").write_text("# Requirement Obligation Graph\n", encoding="utf-8")
            (reference / "spec_test.py").write_text("print('PASS: prior')\n", encoding="utf-8")
            agent = object.__new__(CodeSpec2OnlyReviewAgent)
            agent.config = OnlyReviewConfig(repo_path=repo, task="task", log_root=root / "logs", initial_patch=patch, reference_artifacts=reference, model_name="test")
            agent.log_dir = root / "logs" / "review"
            agent.log_dir.mkdir(parents=True)
            agent.artifacts = make_artifacts(agent.log_dir)
            agent._stage_counts = {}
            agent._initialize_reference_artifacts()
            agent._apply_initial_patch()
            self.assertEqual((repo / "module.py").read_text(encoding="utf-8"), "value = 'prior'\n")
            self.assertEqual(agent.artifacts.spec_test.read_text(encoding="utf-8"), "print('PASS: prior')\n")


if __name__ == "__main__":
    unittest.main()
