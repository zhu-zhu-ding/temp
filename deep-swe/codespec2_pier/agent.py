"""Run the local CodeSpec2 pipeline as an isolated Pier agent.

Only the two agent implementation directories are uploaded to the task
container.  In particular, this adapter never uploads a DeepSWE task directory,
hidden tests, a reference solution, or the FeatureBench ``Agent`` parent.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Keep the host-side custom-agent import offline as well.  This must precede
# Pier's utility import, which imports LiteLLM.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

from dotenv import load_dotenv
from pier.agents.installed.base import BaseInstalledAgent
from pier.agents.network import allowlist_from_urls
from pier.agents.utils import get_api_key_var_names_from_model_name
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist


_DEEP_SWE_ROOT = Path(__file__).resolve().parents[1]
_FEATUREBENCH_ROOT = _DEEP_SWE_ROOT.parent
_CODESPEC2_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "codespec2"
_QUAD_SPEC_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "quad_spec_agent"
_PATCH_FUSE_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "patch_fuse_agent"
_CLARIFY_GPT_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "clarify_gpt_agent"
_ASK_FIRST_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "ask_first_agent"
_SPEC_FIX_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "specfix_agent"
_ARCH_AGENT_SOURCE = _FEATUREBENCH_ROOT / "Agent" / "arch_agent_mini_swe"

_CONTAINER_AGENT_ROOT = "/installed-agent/codespec2-runtime"
_CONTAINER_CODESPEC2_ROOT = f"{_CONTAINER_AGENT_ROOT}/codespec2"
_CONTAINER_QUAD_SPEC_ROOT = f"{_CONTAINER_AGENT_ROOT}/quad_spec_agent"
_CONTAINER_ORACLE_ROOT = f"{_CONTAINER_AGENT_ROOT}/quad_spec_oracle"
_CONTAINER_CLARIFY_GPT_ROOT = f"{_CONTAINER_AGENT_ROOT}/clarify_gpt_agent"
_CONTAINER_ASK_FIRST_ROOT = f"{_CONTAINER_AGENT_ROOT}/ask_first_agent"
_CONTAINER_SPEC_FIX_ROOT = f"{_CONTAINER_AGENT_ROOT}/specfix_agent"
_CONTAINER_PATCH_FUSE_ROOT = f"{_CONTAINER_AGENT_ROOT}/patch_fuse_agent"
_CONTAINER_ARCH_AGENT_ROOT = f"{_CONTAINER_AGENT_ROOT}/arch_agent_mini_swe"
_CONTAINER_LOG_ROOT = "/logs/agent/codespec2"
_CONTAINER_CANDIDATES_ROOT = f"{_CONTAINER_LOG_ROOT}/candidates"
_CONTAINER_RUNTIME_SITE = f"{_CONTAINER_AGENT_ROOT}/python"
_CONTAINER_PYTHON = "python3"
_LOCAL_ENV_FILE = _DEEP_SWE_ROOT / ".env.local"
_ONLY_REVIEW_SOURCE_RUN = _DEEP_SWE_ROOT / "jobs" / "2026-08-19__02-37-37" / "kombu-virtual-queue-dead-letteri__4hLqeP5"
_ONLY_REVIEW_PATCH = _ONLY_REVIEW_SOURCE_RUN / "artifacts" / "model.patch"
_ONLY_REVIEW_ARTIFACTS = _ONLY_REVIEW_SOURCE_RUN / "agent" / "codespec2" / "pier_task_20260819_063747" / "artifacts"
_CONTAINER_ONLY_REVIEW_REFERENCE = f"{_CONTAINER_LOG_ROOT}/only_review_reference"
_RUNTIME_CACHE = _DEEP_SWE_ROOT / ".codespec2_runtime"
_RUNTIME_SITE = Path(
    os.environ.get(
        "CODESPEC2_RUNTIME_SITE_PACKAGES",
        "/home/wpd/miniconda3/envs/openhands/lib/python3.12/site-packages",
    )
)
_RUNTIME_ROOT_DISTS = ("mini-swe-agent", "pyflakes")


def _validate_oracle_source(oracle_source: str) -> str:
    """Fail fast on a typo before any container is started."""

    if oracle_source not in {"gold_patch", "tests"}:
        raise ValueError(f"oracle_source must be 'gold_patch' or 'tests', got {oracle_source!r}")
    return oracle_source


def _match_task_dir(tasks_root: Path, instruction: str, label: str) -> Path:
    """Map the instruction to its unique dataset task directory."""

    from pier.models.task.task import strip_canary

    normalized = " ".join(instruction.split())
    matches = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        text = (task_dir / "instruction.md").read_text(encoding="utf-8", errors="replace")
        if " ".join(strip_canary(text).split()) == normalized:
            matches.append(task_dir)
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError(
            f"{label} task resolution matched {len(matches)} tasks "
            f"({names}); expected exactly one under {tasks_root}."
        )
    return matches[0]


def _stage_tests_dir(task_dir: Path) -> Path:
    """Stage only the standard test code (test.patch + test.sh) for upload.

    grader.py, Dockerfile, and config.json are grading internals and stay
    host-side; the oracle's privileged source is the test code alone.
    """

    tests = task_dir / "tests"
    staged = Path(tempfile.mkdtemp(prefix="oracle-tests-"))
    for name in ("test.patch", "test.sh"):
        source = tests / name
        if source.is_file():
            shutil.copy2(source, staged / name)
    if not any(staged.iterdir()):
        shutil.rmtree(staged, ignore_errors=True)
        raise FileNotFoundError(f"No test.patch/test.sh under {tests}; the oracle stage cannot run.")
    return staged
# CodeSpec2 imports the LocalEnvironment and LiteLLM model path only.  The
# benchmark/dataset stack is optional mini-swe-agent functionality, unrelated
# to task solving, and would add hundreds of MB to every container transfer.
_RUNTIME_EXCLUDED_DISTS = {"datasets"}


class CodeSpec2PierAgent(BaseInstalledAgent):
    """CodeSpec2's spec-agent → implement flow.

    The task checkout remains the regular Pier workspace at ``/app``.  Agent
    artifacts are written under ``/logs/agent/codespec2`` and are not mixed into
    the checkout submitted to the verifier.
    """

    def __init__(
        self,
        spec_agent_step_limit: int = 300,
        implementation_step_limit: int = 300,
        debug_step_limit: int = 300,
        cost_limit: float = 3.0,
        command_timeout: int = 60,
        wall_time_limit_seconds: int = 0,
        mini_swe_version: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Pier creates this custom agent in the host process.  Loading this
        # gitignored, local-only file here makes the normal command concise
        # while keeping credentials out of the task container filesystem and
        # out of source control.  Existing shell variables and explicit
        # --agent-env values intentionally take precedence.
        load_dotenv(_LOCAL_ENV_FILE, override=False)
        kwargs.setdefault("version", "1.0.0")
        super().__init__(*args, **kwargs)
        self.spec_agent_step_limit = int(spec_agent_step_limit)
        self.implementation_step_limit = int(implementation_step_limit)
        self.debug_step_limit = int(debug_step_limit)
        self.cost_limit = float(cost_limit)
        self.command_timeout = int(command_timeout)
        self.wall_time_limit_seconds = int(wall_time_limit_seconds)
        self.mini_swe_version = mini_swe_version
        self._last_run_status: int | None = None

    @staticmethod
    def name() -> str:
        return "codespec2"

    def install_spec(self) -> AgentInstallSpec:
        """Install just the runtime dependencies needed by CodeSpec2.

        Source is deliberately uploaded in :meth:`setup`, rather than copied
        into an image build context.  That keeps a local development checkout
        usable and makes the transfer set explicit and auditable.
        """

        # Do not download anything in the Dockerfile.  The Python runtime is
        # materialized from a local, audited cache during setup().
        root_run = "mkdir -p /installed-agent/codespec2-runtime"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self.version(),
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
            ],
            verification_command="test -d /installed-agent/codespec2-runtime",
        )

    def network_allowlist(self) -> NetworkAllowlist:
        urls = [
            value
            for key in (
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "DEEPSEEK_BASE_URL",
                "DEEPSEEK_API_BASE",
            )
            if (value := self._get_env(key))
        ]
        # The agent runtime is uploaded offline, so only the selected model
        # provider/gateway needs network access.
        default_domains: list[str] = []
        if self.model_name:
            provider = self.model_name.split("/", 1)[0]
            default_domains.extend({
                "deepseek": ["api.deepseek.com"],
                "openai": ["api.openai.com"],
            }.get(provider, []))
        return allowlist_from_urls(urls, default_domains=default_domains)

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_local_sources()
        await super().setup(environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_CODESPEC2_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        runtime_bundle = _ensure_offline_runtime_bundle()
        await environment.upload_dir(runtime_bundle, _CONTAINER_RUNTIME_SITE)
        # Upload only the two declared agent source directories.  Never upload
        # the FeatureBench parent or a DeepSWE task path: either can contain
        # benchmark-only material that the task agent must not inspect.
        await environment.upload_dir(_CODESPEC2_SOURCE, _CONTAINER_CODESPEC2_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("CodeSpec2 requires --model provider/model_name")

        env = self._runtime_env()
        # Do not write directly below self.logs_dir: the nested directory was
        # created inside the bind-mounted container by root and may be
        # unwritable by the host Pier user.  Docker's upload operation handles
        # the destination permissions inside the container.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".codespec2-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )

        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_CODESPEC2_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} '
            '--model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--spec-agent-step-limit {self.spec_agent_step_limit} '
            f'--implementation-step-limit {self.implementation_step_limit} '
            f'--debug-step-limit {self.debug_step_limit} '
            f'--cost-limit {self.cost_limit} '
            f'--timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code

        # DeepSWE verifies the committed delta.  Preserve a partial
        # implementation for evaluation even if one CodeSpec2 stage returned a
        # non-zero status; the status is kept in logs and trial metadata.
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email codespec2@pier.local && "
                "git config user.name CodeSpec2 && git add -A && "
                "(git diff --cached --quiet || git commit -m 'CodeSpec2 solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "CodeSpec2 could not commit the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "codespec2_run_exit_code": result.return_code,
            "codespec2_log_dir": _CONTAINER_LOG_ROOT,
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        # CodeSpec2 emits one mini-swe trajectory per stage.  Pier's native
        # single-trajectory converter is not applicable; retain stage logs as
        # artifacts and report the process status in ``context.metadata``.
        return None

    def _runtime_env(self) -> dict[str, str]:
        env = self.build_process_env(
            {
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
                "PYTHONPATH": f"{_CONTAINER_RUNTIME_SITE}:{_CONTAINER_AGENT_ROOT}",
            }
        )
        # ``build_process_env`` deliberately starts from explicit agent
        # settings.  Forward the credentials/config loaded from .env.local as
        # well; explicit --agent-env values already win through _get_env.
        for key in (
            "MSWEA_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_BASE",
        ):
            if value := self._get_env(key):
                env.setdefault(key, value)
        if "DEEPSEEK_BASE_URL" in env and "DEEPSEEK_API_BASE" not in env:
            env["DEEPSEEK_API_BASE"] = env["DEEPSEEK_BASE_URL"]
        if "OPENAI_BASE_URL" in env and "OPENAI_API_BASE" not in env:
            env["OPENAI_API_BASE"] = env["OPENAI_BASE_URL"]

        if env.get("MSWEA_API_KEY"):
            return env
        try:
            key_vars = get_api_key_var_names_from_model_name(self.model_name or "")
        except ValueError as exc:
            raise ValueError(
                f"Cannot infer credentials for {self.model_name!r}; pass "
                "--agent-env MSWEA_API_KEY=..."
            ) from exc
        missing = [key for key in key_vars if not env.get(key)]
        if missing:
            raise ValueError(
                f"Missing API credential for {self.model_name}: set "
                f"{', '.join(missing)} or MSWEA_API_KEY via --agent-env."
            )
        return env

    @staticmethod
    def _validate_local_sources() -> None:
        required = (
            _CODESPEC2_SOURCE / "run_agent.py",
            _CODESPEC2_SOURCE / "orchestrator.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe_simple" / "repo_understand.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CodeSpec2 local sources are incomplete; expected: " + ", ".join(missing)
            )

    @staticmethod
    def _owner_fix_command(default_user: str | int | None) -> str:
        """Return a safe ownership adjustment for explicitly non-root tasks."""

        if default_user is None or str(default_user) in {"root", "0"}:
            return ""
        owner = shlex.quote(str(default_user))
        return (
            f" && id -u {owner} >/dev/null 2>&1 && "
            f"chown -R {owner} {shlex.quote(_CONTAINER_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}"
        )

    async def _upload_oracle_reference(
        self,
        environment: BaseEnvironment,
        oracle_source: str,
        tasks_root: Path,
        instruction: str,
        label: str,
    ) -> str:
        """Upload the oracle's privileged source; return its run_agent CLI flag.

        ``gold_patch`` uploads the task's ``solution/solution.patch``;
        ``tests`` uploads only ``tests/test.patch`` and ``tests/test.sh``
        (the standard test code; grader.py, Dockerfile, and config.json are
        grading internals and stay host-side).  Either way the upload lands
        outside ``/app`` and outside the collected log tree, and the
        in-container agent reads and deletes it at startup.
        """

        if oracle_source not in {"gold_patch", "tests"}:
            raise ValueError(f"oracle_source must be 'gold_patch' or 'tests', got {oracle_source!r}")
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_ORACLE_ROOT)}{owner_fix}",
        )
        if oracle_source == "tests":
            staged = _stage_tests_dir(_match_task_dir(tasks_root, instruction, label))
            container_tests = f"{_CONTAINER_ORACLE_ROOT}/tests"
            await environment.upload_dir(staged, container_tests)
            shutil.rmtree(staged, ignore_errors=True)
            return f"--tests-dir {shlex.quote(container_tests)}"
        task_dir = _match_task_dir(tasks_root, instruction, label)
        gold_patch = task_dir / "solution" / "solution.patch"
        if not gold_patch.is_file():
            raise FileNotFoundError(
                f"No solution/solution.patch under {task_dir}; the oracle stage "
                "cannot run without the reference patch."
            )
        await environment.upload_file(gold_patch, f"{_CONTAINER_ORACLE_ROOT}/gold.patch")
        return f'--gold-patch {shlex.quote(_CONTAINER_ORACLE_ROOT + "/gold.patch")}'


class QuadSpecPierAgent(CodeSpec2PierAgent):
    """Run 3 specs + ask-first in parallel, red-team, oracle, implement.

    Three independent Mini-SWE spec agents and one requirements clarifier
    (ask-first question style) run concurrently against the read-only
    checkout; a red-team agent merges the clarifier's requirement-level
    questions with the specs' disagreement into one behavioral-fork list; an
    oracle agent that represents the task author answers them from the
    task's reference material, which it must never reveal; the
    implementation agent then modifies ``/app`` with those answers appended
    verbatim to the instruction.

    The oracle's privileged source is configurable (``oracle_source``):
    ``gold_patch`` uploads only that task's ``solution/solution.patch``;
    ``tests`` uploads only ``tests/test.patch`` and ``tests/test.sh`` (the
    standard test code, never grader.py/Dockerfile/config.json).  Either way
    the upload lands outside ``/app`` and outside the collected log tree; the
    in-container agent reads and deletes it at startup and holds the content
    in memory, materializing it as private temporary files only while the
    oracle stage runs - so the reference is on disk during no other stage.
    No other stage is told the path, and nothing else from the task
    directory (hidden tests, environment, grading scripts) is ever uploaded.
    """

    _DEFAULT_DATASET_ROOT = _DEEP_SWE_ROOT / "subtasks" / "loop-validation-10"

    def __init__(
        self,
        spec_step_limit: int = 200,
        review_step_limit: int = 200,
        oracle_source: str = "gold_patch",
        tasks_root: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spec_step_limit = int(spec_step_limit)
        self.review_step_limit = int(review_step_limit)
        self.oracle_source = _validate_oracle_source(oracle_source)
        self.tasks_root = (
            Path(tasks_root).expanduser().resolve()
            if tasks_root
            else self._DEFAULT_DATASET_ROOT / "tasks"
        )

    @staticmethod
    def name() -> str:
        return "quad_spec"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_quad_spec_sources()
        # Call BaseInstalledAgent.setup rather than CodeSpec2PierAgent.setup:
        # this uploads only the new agent and the shared Mini-SWE adapter.
        await BaseInstalledAgent.setup(self, environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_QUAD_SPEC_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        await environment.upload_dir(_ensure_offline_runtime_bundle(), _CONTAINER_RUNTIME_SITE)
        await environment.upload_dir(_QUAD_SPEC_SOURCE, _CONTAINER_QUAD_SPEC_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("QuadSpec requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".quad-spec-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_ORACLE_ROOT)}{owner_fix}",
        )
        reference_flag = await self._upload_oracle_reference(
            environment, self.oracle_source, self.tasks_root, instruction, "QuadSpec"
        )
        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_QUAD_SPEC_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--spec-step-limit {self.spec_step_limit} '
            f'--review-step-limit {self.review_step_limit} '
            f'{reference_flag} '
            f'--implementation-step-limit {self.implementation_step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email quad-spec@pier.local && "
                "git config user.name QuadSpec && git add -A && "
                "(git diff --cached --quiet || git commit -m 'QuadSpec solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "QuadSpec could not finalize the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "quad_spec_run_exit_code": result.return_code,
            "quad_spec_log_dir": _CONTAINER_LOG_ROOT,
        }

    @staticmethod
    def _validate_quad_spec_sources() -> None:
        required = (
            _QUAD_SPEC_SOURCE / "run_agent.py",
            _QUAD_SPEC_SOURCE / "orchestrator.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "QuadSpec local sources are incomplete; expected: " + ", ".join(missing)
            )


class CodeSpec2OnlyReviewPierAgent(CodeSpec2PierAgent):
    """Repair one previous CodeSpec2 attempt from its model patch and specs.

    This mode deliberately uploads only the previous model-generated patch and
    its two public agent artifacts.  It never uploads the old task checkout,
    verifier output, hidden tests, or DeepSWE solution materials.
    """

    def __init__(
        self,
        review_step_limit: int = 300,
        source_run: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        root = Path(source_run).expanduser().resolve() if source_run else _ONLY_REVIEW_SOURCE_RUN
        self.review_step_limit = int(review_step_limit)
        self.only_review_patch = root / "artifacts" / "model.patch"
        artifact_candidates = sorted((root / "agent" / "codespec2").glob("*/artifacts"))
        self.only_review_artifacts = artifact_candidates[-1] if artifact_candidates else root / "agent" / "codespec2" / "artifacts"

    @staticmethod
    def name() -> str:
        return "codespec2_only_review"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_only_review_source()
        await super().setup(environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_ONLY_REVIEW_REFERENCE)}{owner_fix}",
        )
        await environment.upload_file(
            self.only_review_patch,
            f"{_CONTAINER_ONLY_REVIEW_REFERENCE}/initial.patch",
        )
        for name in ("spec.md", "spec_test.py"):
            await environment.upload_file(
                self.only_review_artifacts / name,
                f"{_CONTAINER_ONLY_REVIEW_REFERENCE}/{name}",
            )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("CodeSpec2 only-review requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".codespec2-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )
        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_CODESPEC2_ROOT + "/run_only_review_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--initial-patch {shlex.quote(_CONTAINER_ONLY_REVIEW_REFERENCE + "/initial.patch")} '
            f'--reference-artifacts {shlex.quote(_CONTAINER_ONLY_REVIEW_REFERENCE)} '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--review-step-limit {self.review_step_limit} '
            f'--implementation-step-limit {self.implementation_step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=("set -o pipefail; cd /app && " f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email codespec2@pier.local && "
                "git config user.name CodeSpec2 && git add -A && "
                "(git diff --cached --quiet || git commit -m 'CodeSpec2 only-review solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError("CodeSpec2 only-review could not commit the task workspace: " + (commit.stderr or commit.stdout or "no output"))
        context.metadata = {
            "codespec2_only_review_run_exit_code": result.return_code,
            "codespec2_only_review_log_dir": _CONTAINER_LOG_ROOT,
        }

    def _validate_only_review_source(self) -> None:
        self._validate_local_sources()
        required = (self.only_review_patch, self.only_review_artifacts / "spec.md", self.only_review_artifacts / "spec_test.py")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("CodeSpec2 only-review source is incomplete; expected: " + ", ".join(missing))


class PatchFusionPierAgent(CodeSpec2PierAgent):
    """Fuse four prior Mini-SWE candidate patches into one best patch.

    The candidates are the ``model.patch`` exports of four independent
    baseline attempts.  Per trial the adapter matches the instruction against
    the local dataset's ``instruction.md`` files and uploads that task's
    ``candidate_*.patch`` files read-only into the container.  Only the patch
    files are uploaded: never verifier output, hidden tests, or the old task
    checkouts the patches came from.
    """

    _DEFAULT_DATASET_ROOT = _DEEP_SWE_ROOT / "subtasks" / "loop-validation-10"

    def __init__(
        self,
        mode: str = "fusion",
        step_limit: int = 300,
        candidates_root: str | Path | None = None,
        tasks_root: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if mode not in {"baseline", "fusion", "quad_fuse"}:
            raise ValueError(f"mode must be 'baseline', 'fusion' or 'quad_fuse', got {mode!r}")
        self.mode = mode
        self.step_limit = int(step_limit)
        dataset_root = self._DEFAULT_DATASET_ROOT
        self.candidates_root = (
            Path(candidates_root).expanduser().resolve()
            if candidates_root
            else dataset_root / "candidates"
        )
        self.tasks_root = (
            Path(tasks_root).expanduser().resolve() if tasks_root else dataset_root / "tasks"
        )

    @staticmethod
    def name() -> str:
        return "patch_fusion"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_patch_fuse_sources()
        await BaseInstalledAgent.setup(self, environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_PATCH_FUSE_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        await environment.upload_dir(_ensure_offline_runtime_bundle(), _CONTAINER_RUNTIME_SITE)
        await environment.upload_dir(_PATCH_FUSE_SOURCE, _CONTAINER_PATCH_FUSE_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("PatchFusion requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".patch-fusion-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )

        candidates_flag = ""
        if self.mode in {"fusion", "quad_fuse"}:
            candidates = self._resolve_task_candidates(instruction)
            owner_fix = self._owner_fix_command(environment.default_user)
            await self.exec_as_root(
                environment,
                f"mkdir -p {shlex.quote(_CONTAINER_CANDIDATES_ROOT)}{owner_fix}",
            )
            for path in candidates:
                await environment.upload_file(path, f"{_CONTAINER_CANDIDATES_ROOT}/{path.name}")
            candidates_flag = f" --candidates-dir {shlex.quote(_CONTAINER_CANDIDATES_ROOT)}"

        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_PATCH_FUSE_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--mode {self.mode}{candidates_flag} '
            f'--step-limit {self.step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email patch-fusion@pier.local && "
                "git config user.name PatchFusion && git add -A && "
                "(git diff --cached --quiet || git commit -m 'PatchFusion solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "PatchFusion could not finalize the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "patch_fusion_run_exit_code": result.return_code,
            "patch_fusion_log_dir": _CONTAINER_LOG_ROOT,
        }

    def _resolve_task_candidates(self, instruction: str) -> list[Path]:
        """Map the instruction to its dataset task and return its candidates."""

        from pier.models.task.task import strip_canary

        normalized = " ".join(instruction.split())
        matches = []
        for task_dir in sorted(path for path in self.tasks_root.iterdir() if path.is_dir()):
            text = (task_dir / "instruction.md").read_text(encoding="utf-8", errors="replace")
            if " ".join(strip_canary(text).split()) == normalized:
                matches.append(task_dir)
        if len(matches) != 1:
            names = [path.name for path in matches]
            raise ValueError(
                f"PatchFusion candidate resolution matched {len(matches)} tasks "
                f"({names}); expected exactly one under {self.tasks_root}."
            )
        candidates = sorted((self.candidates_root / matches[0].name).glob("candidate_*.patch"))
        if not candidates:
            raise FileNotFoundError(
                f"No candidate_*.patch under {self.candidates_root / matches[0].name}."
            )
        return candidates

    @staticmethod
    def _validate_patch_fuse_sources() -> None:
        required = (
            _PATCH_FUSE_SOURCE / "run_agent.py",
            _PATCH_FUSE_SOURCE / "orchestrator.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "PatchFusion local sources are incomplete; expected: " + ", ".join(missing)
            )


class ClarifyGptPierAgent(PatchFusionPierAgent):
    """ClarifyGPT baseline built on the patch-fusion candidate flow.

    Four prior Mini-SWE candidate patches are uploaded read-only; a clarify
    agent (the official ClarifyGPT askcq prompt, R1/R2 variant) compares the
    candidates' understandings and asks at most ten clarifying questions; an
    oracle user that represents the task author answers them from privileged
    reference material (``oracle_source``: the gold patch or the project's
    standard test code), which it must never reveal; the implementation
    agent then regenerates the solution from the requirement with those
    answers appended verbatim.  The implementation stage never sees the
    candidates and no verifier output or hidden test is ever uploaded.

    Subclasses PatchFusionPierAgent to reuse instruction-to-task candidate
    resolution; ``setup`` and ``run`` are overridden wholesale, so the parent
    ``mode`` is never used at runtime ("fusion" only satisfies its validator).
    """

    _DEFAULT_DATASET_ROOT = _DEEP_SWE_ROOT / "subtasks" / "clarify-gpt-python-34"

    def __init__(
        self,
        clarify_step_limit: int = 200,
        implementation_step_limit: int = 300,
        oracle_source: str = "gold_patch",
        candidates_root: str | Path | None = None,
        tasks_root: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mode="fusion",
            step_limit=int(implementation_step_limit),
            candidates_root=candidates_root,
            tasks_root=tasks_root if tasks_root else self._DEFAULT_DATASET_ROOT / "tasks",
            *args,
            **kwargs,
        )
        self.clarify_step_limit = int(clarify_step_limit)
        self.implementation_step_limit = int(implementation_step_limit)
        self.oracle_source = _validate_oracle_source(oracle_source)

    @staticmethod
    def name() -> str:
        return "clarify_gpt"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_clarify_gpt_sources()
        # Call BaseInstalledAgent.setup rather than PatchFusionPierAgent.setup:
        # this uploads the clarify_gpt agent, not patch_fuse_agent.
        await BaseInstalledAgent.setup(self, environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_CLARIFY_GPT_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        await environment.upload_dir(_ensure_offline_runtime_bundle(), _CONTAINER_RUNTIME_SITE)
        await environment.upload_dir(_CLARIFY_GPT_SOURCE, _CONTAINER_CLARIFY_GPT_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("ClarifyGPT requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".clarify-gpt-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )

        candidates = self._resolve_task_candidates(instruction)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_CANDIDATES_ROOT)} "
            f"{shlex.quote(_CONTAINER_ORACLE_ROOT)}{owner_fix}",
        )
        for path in candidates:
            await environment.upload_file(path, f"{_CONTAINER_CANDIDATES_ROOT}/{path.name}")
        reference_flag = await self._upload_oracle_reference(
            environment, self.oracle_source, self.tasks_root, instruction, "ClarifyGpt"
        )

        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_CLARIFY_GPT_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--candidates-dir {shlex.quote(_CONTAINER_CANDIDATES_ROOT)} '
            f'{reference_flag} '
            f'--clarify-step-limit {self.clarify_step_limit} '
            f'--implementation-step-limit {self.implementation_step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email clarify-gpt@pier.local && "
                "git config user.name ClarifyGPT && git add -A && "
                "(git diff --cached --quiet || git commit -m 'ClarifyGPT solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "ClarifyGPT could not finalize the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "clarify_gpt_run_exit_code": result.return_code,
            "clarify_gpt_log_dir": _CONTAINER_LOG_ROOT,
        }

    def _resolve_task_gold_patch(self, instruction: str) -> Path:
        """Map the instruction to its dataset task and return its solution patch."""

        task_dir = _match_task_dir(self.tasks_root, instruction, "ClarifyGPT")
        gold_patch = task_dir / "solution" / "solution.patch"
        if not gold_patch.is_file():
            raise FileNotFoundError(
                f"No solution/solution.patch under {task_dir}; the oracle stage "
                "cannot run without the reference patch."
            )
        return gold_patch

    @staticmethod
    def _validate_clarify_gpt_sources() -> None:
        required = (
            _CLARIFY_GPT_SOURCE / "run_agent.py",
            _CLARIFY_GPT_SOURCE / "orchestrator.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "ClarifyGPT local sources are incomplete; expected: " + ", ".join(missing)
            )


class AskFirstPierAgent(CodeSpec2PierAgent):
    """Ask-first clarification baseline: questions from the requirement alone.

    A clarify agent reads the task text (and may read the repository to
    settle apparent ambiguities) and asks at most ten clarifying questions.
    If any question exists, an oracle user that represents the task author
    answers them from privileged reference material (``oracle_source``:
    the gold patch or the project's standard test code), which it must never
    reveal, and the implementation agent receives the task text with those
    answers appended verbatim; if none exists, the implementation agent
    solves the original task directly.  No candidate patches,
    specifications, verifier output, or hidden test is ever uploaded - this
    isolates the divergence signal by removing it entirely.
    """

    _DEFAULT_DATASET_ROOT = _DEEP_SWE_ROOT / "subtasks" / "clarify-gpt-python-34"

    def __init__(
        self,
        ask_step_limit: int = 200,
        implementation_step_limit: int = 300,
        oracle_source: str = "gold_patch",
        tasks_root: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ask_step_limit = int(ask_step_limit)
        self.implementation_step_limit = int(implementation_step_limit)
        self.oracle_source = _validate_oracle_source(oracle_source)
        self.tasks_root = (
            Path(tasks_root).expanduser().resolve()
            if tasks_root
            else self._DEFAULT_DATASET_ROOT / "tasks"
        )

    @staticmethod
    def name() -> str:
        return "ask_first"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_ask_first_sources()
        # Call BaseInstalledAgent.setup rather than CodeSpec2PierAgent.setup:
        # this uploads only the new agent and the shared Mini-SWE adapter.
        await BaseInstalledAgent.setup(self, environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_ASK_FIRST_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        await environment.upload_dir(_ensure_offline_runtime_bundle(), _CONTAINER_RUNTIME_SITE)
        await environment.upload_dir(_ASK_FIRST_SOURCE, _CONTAINER_ASK_FIRST_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("AskFirst requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".ask-first-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )
        reference_flag = await self._upload_oracle_reference(
            environment, self.oracle_source, self.tasks_root, instruction, "AskFirst"
        )
        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_ASK_FIRST_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'{reference_flag} '
            f'--ask-step-limit {self.ask_step_limit} '
            f'--implementation-step-limit {self.implementation_step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email ask-first@pier.local && "
                "git config user.name AskFirst && git add -A && "
                "(git diff --cached --quiet || git commit -m 'AskFirst solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "AskFirst could not finalize the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "ask_first_run_exit_code": result.return_code,
            "ask_first_log_dir": _CONTAINER_LOG_ROOT,
        }

    def _resolve_task_gold_patch(self, instruction: str) -> Path:
        """Map the instruction to its dataset task and return its solution patch."""

        task_dir = _match_task_dir(self.tasks_root, instruction, "AskFirst")
        gold_patch = task_dir / "solution" / "solution.patch"
        if not gold_patch.is_file():
            raise FileNotFoundError(
                f"No solution/solution.patch under {task_dir}; the oracle stage "
                "cannot run without the reference patch."
            )
        return gold_patch

    @staticmethod
    def _validate_ask_first_sources() -> None:
        required = (
            _ASK_FIRST_SOURCE / "run_agent.py",
            _ASK_FIRST_SOURCE / "orchestrator.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "AskFirst local sources are incomplete; expected: " + ", ".join(missing)
            )


class SpecFixPierAgent(PatchFusionPierAgent):
    """SpecFix baseline over the four reused candidate patches.

    A test-generation agent writes behavioral probes plus explicit-assertion
    tests for the requirement; the orchestrator applies each candidate patch,
    runs the suites, and clusters candidates by behavioral signature
    (differential testing).  When the clusters disagree (entropy or
    consistency), up to three official repair rounds follow: fault
    localization against a reference patch, requirement repair, and a fresh
    regenerated patch that joins the pool; the finally selected majority
    cluster's representative patch is applied as the solution.  Patches that
    do not apply or whose suites cannot run are excluded - the agent's own
    tests do the excluding.  No verifier output, hidden test, or reference
    solution is ever uploaded.

    Subclasses PatchFusionPierAgent to reuse instruction-to-task candidate
    resolution; ``setup`` and ``run`` are overridden wholesale, so the parent
    ``mode`` is never used at runtime ("fusion" only satisfies its validator).
    """

    _DEFAULT_DATASET_ROOT = _DEEP_SWE_ROOT / "subtasks" / "clarify-gpt-python-34"

    def __init__(
        self,
        repair_attempts: int = 3,
        test_gen_step_limit: int = 150,
        program_repair_step_limit: int = 200,
        localize_step_limit: int = 150,
        requirement_repair_step_limit: int = 150,
        regen_step_limit: int = 250,
        implement_step_limit: int = 250,
        candidates_root: str | Path | None = None,
        tasks_root: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mode="fusion",
            step_limit=int(implement_step_limit),
            candidates_root=candidates_root,
            tasks_root=tasks_root if tasks_root else self._DEFAULT_DATASET_ROOT / "tasks",
            *args,
            **kwargs,
        )
        self.repair_attempts = int(repair_attempts)
        self.test_gen_step_limit = int(test_gen_step_limit)
        self.program_repair_step_limit = int(program_repair_step_limit)
        self.localize_step_limit = int(localize_step_limit)
        self.requirement_repair_step_limit = int(requirement_repair_step_limit)
        self.regen_step_limit = int(regen_step_limit)
        self.spec_fix_implement_step_limit = int(implement_step_limit)

    @staticmethod
    def name() -> str:
        return "spec_fix"

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_spec_fix_sources()
        # Call BaseInstalledAgent.setup rather than PatchFusionPierAgent.setup:
        # this uploads the specfix agent, not patch_fuse_agent.
        await BaseInstalledAgent.setup(self, environment)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_SPEC_FIX_ROOT)} "
            f"{shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)} "
            f"{shlex.quote(_CONTAINER_LOG_ROOT)}{owner_fix}",
        )
        await environment.upload_dir(_ensure_offline_runtime_bundle(), _CONTAINER_RUNTIME_SITE)
        await environment.upload_dir(_SPEC_FIX_SOURCE, _CONTAINER_SPEC_FIX_ROOT)
        await environment.upload_dir(_ARCH_AGENT_SOURCE, _CONTAINER_ARCH_AGENT_ROOT)
        await self.exec_as_root(
            environment,
            f"chmod -R a-w {shlex.quote(_CONTAINER_ARCH_AGENT_ROOT)} "
            f"{shlex.quote(_CONTAINER_RUNTIME_SITE)}{owner_fix}",
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("SpecFix requires --model provider/model_name")
        env = self._runtime_env()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".spec-fix-task", delete=True
        ) as instruction_file:
            instruction_file.write(instruction)
            instruction_file.flush()
            await environment.upload_file(
                instruction_file.name, f"{_CONTAINER_LOG_ROOT}/instruction.md"
            )

        candidates = self._resolve_task_candidates(instruction)
        owner_fix = self._owner_fix_command(environment.default_user)
        await self.exec_as_root(
            environment,
            f"mkdir -p {shlex.quote(_CONTAINER_CANDIDATES_ROOT)}{owner_fix}",
        )
        for path in candidates:
            await environment.upload_file(path, f"{_CONTAINER_CANDIDATES_ROOT}/{path.name}")

        python_command = (
            f'{shlex.quote(_CONTAINER_PYTHON)} '
            f'{shlex.quote(_CONTAINER_SPEC_FIX_ROOT + "/run_agent.py")} '
            '--workspace /app '
            f'--task-file {shlex.quote(_CONTAINER_LOG_ROOT + "/instruction.md")} '
            '--task-id pier_task '
            f'--model {shlex.quote(self.model_name)} --model-class litellm '
            f'--log-root {shlex.quote(_CONTAINER_LOG_ROOT)} '
            f'--candidates-dir {shlex.quote(_CONTAINER_CANDIDATES_ROOT)} '
            f'--repair-attempts {self.repair_attempts} '
            f'--test-gen-step-limit {self.test_gen_step_limit} '
            f'--program-repair-step-limit {self.program_repair_step_limit} '
            f'--localize-step-limit {self.localize_step_limit} '
            f'--requirement-repair-step-limit {self.requirement_repair_step_limit} '
            f'--regen-step-limit {self.regen_step_limit} '
            f'--implement-step-limit {self.spec_fix_implement_step_limit} '
            f'--cost-limit {self.cost_limit} --timeout {self.command_timeout} '
            f'--wall-time-limit-seconds {self.wall_time_limit_seconds}'
        )
        result = await environment.exec(
            command=(
                "set -o pipefail; cd /app && "
                f"{python_command} 2>&1 | tee {_CONTAINER_LOG_ROOT}/run.log"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        self._last_run_status = result.return_code
        commit = await environment.exec(
            command=(
                "git config --global --add safe.directory /app && cd /app && "
                "git config user.email spec-fix@pier.local && "
                "git config user.name SpecFix && git add -A && "
                "(git diff --cached --quiet || git commit -m 'SpecFix solution')"
            ),
            user=environment.default_user,
            env=environment.agent_process_env(env),
        )
        if commit.return_code != 0:
            raise RuntimeError(
                "SpecFix could not finalize the task workspace: "
                f"{commit.stderr or commit.stdout or 'no output'}"
            )
        context.metadata = {
            "spec_fix_run_exit_code": result.return_code,
            "spec_fix_log_dir": _CONTAINER_LOG_ROOT,
        }

    @staticmethod
    def _validate_spec_fix_sources() -> None:
        required = (
            _SPEC_FIX_SOURCE / "run_agent.py",
            _SPEC_FIX_SOURCE / "orchestrator.py",
            _SPEC_FIX_SOURCE / "differential.py",
            _ARCH_AGENT_SOURCE / "arch_agent_mini_swe" / "mini_swe.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "SpecFix local sources are incomplete; expected: " + ", ".join(missing)
            )


def _ensure_offline_runtime_bundle() -> Path:
    """Create a host-cached, Python-3.12-compatible mini-swe runtime bundle.

    The bundle is assembled exclusively from the user's existing OpenHands
    environment, then copied into a Pier task container with ``upload_dir``.
    It is not a task mount and contains neither benchmark data nor source from
    the target repository.  A manifest makes regeneration deterministic when
    the installed mini-swe-agent version changes.
    """

    import importlib.metadata as metadata
    import json
    from packaging.requirements import Requirement

    if not _RUNTIME_SITE.is_dir():
        raise FileNotFoundError(
            "Offline CodeSpec2 runtime source is missing: "
            f"{_RUNTIME_SITE}. Set CODESPEC2_RUNTIME_SITE_PACKAGES to a Python "
            "3.12 site-packages directory containing mini-swe-agent and pyflakes."
        )

    # AgentFactory imports this module from Pier's own environment, not from
    # OpenHands.  Restrict distribution discovery to the declared source site.
    distributions = {
        dist.metadata["Name"].lower().replace("_", "-"): dist
        for dist in metadata.distributions(path=[str(_RUNTIME_SITE)])
        if dist.metadata.get("Name")
    }
    required = _runtime_distribution_closure(distributions, Requirement)
    manifest = {
        "schema": 1,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "distributions": {
            name: distributions[name].version for name in sorted(required)
        },
    }
    manifest_path = _RUNTIME_CACHE / "manifest.json"
    if manifest_path.is_file():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
                return _RUNTIME_CACHE / "site-packages"
        except json.JSONDecodeError:
            pass

    staging = _RUNTIME_CACHE.with_name(f"{_RUNTIME_CACHE.name}.building")
    shutil.rmtree(staging, ignore_errors=True)
    target_site = staging / "site-packages"
    target_site.mkdir(parents=True)
    for name in sorted(required):
        dist = distributions[name]
        for relative_path in dist.files or []:
            source = (Path(dist.locate_file(relative_path))).resolve()
            try:
                source.relative_to(_RUNTIME_SITE.resolve())
            except ValueError:
                continue
            if not source.is_file():
                continue
            target = target_site / source.relative_to(_RUNTIME_SITE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(_RUNTIME_CACHE, ignore_errors=True)
    staging.replace(_RUNTIME_CACHE)
    return _RUNTIME_CACHE / "site-packages"


def _runtime_distribution_closure(distributions: dict[str, Any], requirement_type: Any) -> set[str]:
    """Return the mandatory dependency closure needed by mini's LiteLLM path."""

    def normalize(name: str) -> str:
        return name.lower().replace("_", "-")

    pending = [normalize(name) for name in _RUNTIME_ROOT_DISTS]
    required: set[str] = set()
    while pending:
        name = pending.pop()
        if name in required or name in _RUNTIME_EXCLUDED_DISTS:
            continue
        distribution = distributions.get(name)
        if distribution is None:
            raise RuntimeError(
                f"Offline CodeSpec2 runtime is missing required distribution {name!r} "
                f"under {_RUNTIME_SITE}."
            )
        required.add(name)
        for raw_requirement in distribution.requires or []:
            requirement = requirement_type(raw_requirement)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = normalize(requirement.name)
            if dependency not in _RUNTIME_EXCLUDED_DISTS:
                pending.append(dependency)
    return required
