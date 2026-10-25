"""Pure mini-swe-agent reproduction of the RTADev role workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

from featurebench.infer.agents.mini_swe_agent import MiniSweAgent


PRODUCT_MANAGER_PROMPT = """
You are RTADev's Product Manager and Software Requirements Analyst. Analyze the
initial requirement and produce the functional requirements section of an SRS.
Focus on observable behavior, inputs/outputs, compatibility, error cases, and
acceptance criteria. Do not overthink non-functional requirements.

Output format:
[OUTPUT]
## general overview of the project
## software functional requirements
[/OUTPUT]

For this repository repair, do not create code or a new project. The original
task description is the system description. Save the complete result to the
required artifact path before submitting.
"""

ARCHITECT_PROMPT = """
You are RTADev's Architect. Design a concise, complete, and usable architecture
that satisfies the functional requirements. Identify the implementation approach,
affected existing files, data structures and interfaces, imports, callers, and
module relationships. Use a Mermaid class or flow diagram when it clarifies
relationships.

Output format:
[CONTENT]
"Implementation approach": "...",
"File list": ["existing/affected/file.py"],
"Data structures and interfaces": "...",
[/CONTENT]

For this repository repair, inspect the existing repository and minimize the
change surface. Do not create a demo application and do not modify code in this
stage. Save the complete architecture output to the required artifact path.
"""

PROJECT_MANAGER_PROMPT = """
You are RTADev's Project Manager. Break down implementation work according to
the functional requirements and architecture. Analyze dependencies, start from
prerequisite modules, name concrete existing files, required imports/callers,
and focused validation commands.

Output format:
[CONTENT]
"Logic Analysis": [["path", "classes/functions and dependency reasoning"]],
"Task list": ["ordered repair steps"],
"Shared Knowledge": "cross-module contracts and constraints",
[/CONTENT]

For this repository repair, do not create a new project or modify code. Save
the complete ordered repair plan to the required artifact path.
"""

PROGRAMMER_PROMPT = """
You are RTADev's Programmer. Write complete, elegant, modular, maintainable
code according to the architecture and task plan. Follow the design; do not
leave TODOs, pass statements, placeholders, or merely describe code. Check that
all necessary classes, methods, functions, imports, and dependencies exist.

RTADev alignment requirements:
- Check the implementation against the architecture's interfaces and module
  relationships.
- Check the implementation against the task plan's dependency order and shared
  knowledge.
- Repair every mismatch before submitting.

For this repository repair, inspect and edit `/testbed` directly. Preserve
unrelated behavior, implement the task, and run focused repository tests. Save
brief implementation notes to the required artifact path after producing the
working patch.
"""

class RTADevMiniSweOnlyAgent(MiniSweAgent):
    """Use one official mini-swe agent to execute RTADev's original role sequence."""

    @property
    def name(self) -> str:
        return "rtadev_agent_mini_swe_only"

    @property
    def install_script(self) -> str:
        return MiniSweAgent.install_script.fget(self)

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(container, "mkdir -p /agent-logs/rtadev_mini_swe_only", log_file=log_file)
        return True

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        """Write the long multi-stage workflow inside the container before execution."""
        if not super().pre_run_setup(container, instance, log_file):
            return False
        workflow = self._build_workflow(instance.problem_statement)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
                tmp.write(workflow)
                temp_path = Path(tmp.name)
            self.cm.copy_to_container(container, temp_path, "/agent-logs/rtadev_mini_swe_only/workflow.sh")
            exit_code, output = self.cm.exec_command(
                container,
                "chmod 700 /agent-logs/rtadev_mini_swe_only/workflow.sh",
                log_file=log_file,
            )
            if exit_code != 0:
                self.logger.warning("Failed to prepare RTADev workflow: %s", output)
                return False
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _copy_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/rtadev_mini_swe_only",
                log_dir / "rtadev_mini_swe_only",
            )
        except Exception:
            pass
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/rtadev_mini_swe_only_output.log",
                log_dir / "rtadev_mini_swe_only_output.log",
            )
        except Exception:
            pass

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_logs(container, log_file)
        return super().post_run_hook(container, log_file)

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_logs(container, log_file)
        super().failure_hook(container, log_file)

    def _stage_command(self, stage: str, prompt: str, step_limit: int) -> str:
        """Create an unmodified MiniSweAgent invocation with a stage-specific task."""
        command = MiniSweAgent.get_run_command(self, prompt)
        # Stage budgets are independent of the top-level CLI budget.
        command = command.replace(
            f" -c agent.step_limit={self._kwargs.get('max_iters')}",
            f" -c agent.step_limit={step_limit}",
            1,
        )
        if f"agent.step_limit={step_limit}" not in command:
            if " -c mini.yaml" in command:
                command = command.replace(
                    " -c mini.yaml",
                    f" -c mini.yaml -c agent.step_limit={step_limit}",
                    1,
                )
            else:
                command = command.replace(
                    " -y --exit-immediately",
                    f" -y --exit-immediately -c mini.yaml -c agent.step_limit={step_limit}",
                    1,
                )
        return command.replace(
            "tee /agent-logs/mini_swe_agent_output.log",
            f"tee /agent-logs/rtadev_mini_swe_only/{stage}.log",
            1,
        )

    def _build_workflow(self, instruction: str) -> str:
        artifacts = "/agent-logs/rtadev_mini_swe_only"
        task_input = f"""
INPUT TASK DESCRIPTION (problem_statement):
{instruction}
"""
        adaptation = f"""
FeatureBench adaptation: this is an existing-repository repair. Do not create a
separate demo project. Keep code changes in `/testbed`; role artifacts belong
only in `{artifacts}`.
"""
        prompts = [
            (
                "01_product_manager",
                PRODUCT_MANAGER_PROMPT
                + task_input
                + adaptation
                + f"""
STAGE INPUT ARTIFACTS: none.
STAGE OUTPUT ARTIFACT PATH: `{artifacts}/prd.md`
Act only as Product Manager. Do not modify code. Save complete output at the
stage output path, then submit.
""",
                100,
            ),
            (
                "02_architect",
                ARCHITECT_PROMPT
                + task_input
                + adaptation
                + f"""
STAGE INPUT ARTIFACT PATHS:
- Product Manager: `{artifacts}/prd.md`
STAGE OUTPUT ARTIFACT PATH: `{artifacts}/architecture.md`
Read all stage inputs. Act only as Architect. Inspect `/testbed` if needed, do
not modify code, save complete output at the stage output path, then submit.
""",
                100,
            ),
            (
                "03_project_manager",
                PROJECT_MANAGER_PROMPT
                + task_input
                + adaptation
                + f"""
STAGE INPUT ARTIFACT PATHS:
- Product Manager: `{artifacts}/prd.md`
- Architect: `{artifacts}/architecture.md`
STAGE OUTPUT ARTIFACT PATH: `{artifacts}/task_plan.md`
Read all stage inputs. Act only as Project Manager. Inspect `/testbed` if
needed, do not modify code, save complete output at the stage output path, then
submit.
""",
                100,
            ),
            (
                "04_programmer",
                PROGRAMMER_PROMPT
                + task_input
                + adaptation
                + f"""
STAGE INPUT ARTIFACT PATHS:
- Product Manager: `{artifacts}/prd.md`
- Architect: `{artifacts}/architecture.md`
- Project Manager: `{artifacts}/task_plan.md`
STAGE OUTPUT ARTIFACT PATH: `{artifacts}/programmer.md`
Read all stage inputs. Act only as Programmer. Modify `/testbed`, run focused
tests, save notes at the stage output path, then submit.
""",
                200,
            ),
        ]
        commands = [self._stage_command(stage, prompt, step_limit) for stage, prompt, step_limit in prompts]
        # GPT gateways can emit prose-only turns unless tool use is mandatory.
        shell_preamble = "set -e"
        if self._get_model_name().lower().startswith("openai/gpt-"):
            shell_preamble += "\nexport MSWEA_REQUIRE_TOOL_CALLS=required"
        return shell_preamble + "\n" + "\n\n".join(commands)

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        # Keep the host docker-exec argument short; workflow content is copied in
        # pre_run_setup and runs serially inside the task container. The merged
        # transcript preserves all stage costs and reasoning for later analysis.
        return (
            "set -o pipefail; "
            "bash /agent-logs/rtadev_mini_swe_only/workflow.sh 2>&1 | "
            "tee /agent-logs/rtadev_mini_swe_only_output.log"
        )
