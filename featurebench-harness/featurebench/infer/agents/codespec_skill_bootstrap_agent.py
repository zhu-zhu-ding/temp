"""FeatureBench adapter for leakage-resistant CodeSpec skill bootstrapping."""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path

from featurebench.infer.agents.mini_swe_agent import MiniSweAgent


class CodeSpecSkillBootstrapAgent(MiniSweAgent):
    """Train reusable CodeSpec skills without producing a benchmark patch."""

    _WRAPPER_SETUP_COMMAND = r'''
set -e
VENV_DIR="${VENV_DIR:-/opt/mini-swe-agent-venv}"
test -x "$VENV_DIR/bin/python"
test -d /host_arch_swe_agent/codespec_skill_bootstrap
test -d /host_arch_swe_agent/arch_agent_mini_swe/arch_agent_mini_swe
rm -rf /installed-agent/codespec_skill_bootstrap /installed-agent/arch_agent_mini_swe
cp -a /host_arch_swe_agent/codespec_skill_bootstrap /installed-agent/
cp -a /host_arch_swe_agent/arch_agent_mini_swe /installed-agent/
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -q pyflakes "litellm[proxy]"
"$VENV_DIR/bin/python" -m py_compile /installed-agent/codespec_skill_bootstrap/*.py /installed-agent/arch_agent_mini_swe/arch_agent_mini_swe/*.py
test -f /installed-agent/codespec_skill_bootstrap/spec_generation_skill.md
test -f /installed-agent/codespec_skill_bootstrap/review_rubric_skill.md
command -v bwrap >/dev/null 2>&1
'''

    @property
    def name(self) -> str:
        return "codespec_skill_bootstrap"

    @property
    def artifact_only(self) -> bool:
        """FeatureBench must retain artifacts, not interpret an empty git diff as failure."""
        return True

    @property
    def install_script(self) -> str:
        base_script = super().install_script
        return (
            base_script
            + f'''

echo "Installing CodeSpec skill bootstrap wrapper..."
if ! command -v git >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1 || ! command -v bwrap >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git ripgrep bubblewrap
fi
{self._WRAPPER_SETUP_COMMAND}
echo "CodeSpec skill bootstrap wrapper installed"
'''
        )

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        if not super().pre_run_setup(container, instance, log_file):
            raise RuntimeError("Failed to prepare CodeSpec skill bootstrap task payload")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(instance.problem_statement)
                tmp.write("\n")
                temp_path = Path(tmp.name)
            exit_code, output = self.cm.exec_command(
                container,
                "mkdir -p /agent-logs/codespec_skill_bootstrap /bootstrap-skills",
                log_file=log_file,
            )
            if exit_code != 0:
                raise RuntimeError(output)
            self.cm.copy_to_container(
                container, temp_path, "/agent-logs/codespec_skill_bootstrap/requirement.md"
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return True

    def pre_run_hook(self, container, log_file: Path) -> bool:
        exit_code, _ = self.cm.exec_command(
            container,
            "mkdir -p /agent-logs/codespec_skill_bootstrap /bootstrap-skills",
            log_file=log_file,
        )
        return exit_code == 0

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        model_name = self._get_model_name()
        loops = str(self.env_vars.get("CODESPEC_SKILL_BOOTSTRAP_LOOPS") or "1")
        teacher_steps = str(self.env_vars.get("CODESPEC_SKILL_BOOTSTRAP_TEACHER_ITERS") or "160")
        student_steps = str(self.env_vars.get("CODESPEC_SKILL_BOOTSTRAP_STUDENT_ITERS") or "200")
        reviewer_steps = str(self.env_vars.get("CODESPEC_SKILL_BOOTSTRAP_REVIEWER_ITERS") or "160")
        return (
            "set -o pipefail; cd /testbed; "
            "test -f /installed-agent/codespec_skill_bootstrap/run_agent.py || "
            "{ echo 'Missing CodeSpec skill bootstrap agent' >&2; exit 1; }; "
            "PYTHONPATH=/installed-agent:$PYTHONPATH PYTHONUNBUFFERED=1 "
            '"$MINI_SWE_AGENT_PYTHON" -u -m codespec_skill_bootstrap.run_agent '
            "--workspace /testbed "
            "--requirement-file /agent-logs/codespec_skill_bootstrap/requirement.md "
            "--task-id bootstrap "
            f"--model {shlex.quote(model_name)} --model-class litellm "
            f"--loops {shlex.quote(loops)} "
            "--log-root /agent-logs/codespec_skill_bootstrap "
            "--skill-root /bootstrap-skills "
            f"--teacher-step-limit {shlex.quote(teacher_steps)} "
            f"--student-step-limit {shlex.quote(student_steps)} "
            f"--reviewer-step-limit {shlex.quote(reviewer_steps)} "
            "2>&1 | tee /agent-logs/codespec_skill_bootstrap_output.log; "
            "exit ${PIPESTATUS[0]}"
        )

    def post_run_hook(self, container, log_file: Path) -> bool:
        exit_code, output = self.cm.exec_command(
            container,
            "git -C /testbed diff --quiet && git -C /testbed diff --cached --quiet",
            log_file=log_file,
        )
        self._copy_bootstrap_logs(container, log_file)
        if exit_code != 0:
            self.logger.error("Artifact-only bootstrap agent modified /testbed: %s", output)
            return False
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_bootstrap_logs(container, log_file)
        super().failure_hook(container, log_file)

    def _copy_bootstrap_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        for source, destination in (
            ("/agent-logs/codespec_skill_bootstrap", log_dir / "codespec_skill_bootstrap"),
            ("/agent-logs/codespec_skill_bootstrap_output.log", log_dir / "codespec_skill_bootstrap_output.log"),
            ("/bootstrap-skills", log_dir / "codespec_skill_bootstrap_skills"),
        ):
            try:
                self.cm.copy_from_container(container, source, destination)
            except Exception:
                pass
