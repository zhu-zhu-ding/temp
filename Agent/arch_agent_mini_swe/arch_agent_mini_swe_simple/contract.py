from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROLE_NAMES = {
    "core_behavior",
    "public_api",
    "binding",
    "caller_compat",
    "registration",
    "serialization",
    "configuration",
    "error_handling",
    "state_lifecycle",
    "compatibility",
    "validation",
    "repo_specific",
}


@dataclass(frozen=True)
class ContractCheck:
    passed: bool
    output: str


def check_design_contract(contract_path: Path, repo_path: Path) -> ContractCheck:
    data, error = _load_json(contract_path)
    if error:
        return ContractCheck(False, error)

    errors: list[str] = []
    warnings: list[str] = []

    if not _nonempty_str(data.get("task_summary")):
        errors.append("design_contract.task_summary must be a non-empty string.")

    core_interfaces = data.get("core_interfaces")
    if not isinstance(core_interfaces, list) or not core_interfaces:
        errors.append("design_contract.core_interfaces must contain at least one interface.")
    else:
        for idx, item in enumerate(core_interfaces):
            if not isinstance(item, dict):
                errors.append(f"core_interfaces[{idx}] must be an object.")
                continue
            if not _nonempty_str(item.get("symbol")):
                errors.append(f"core_interfaces[{idx}].symbol must be non-empty.")
            _check_files_field(repo_path, item, f"core_interfaces[{idx}]", errors, warnings)

    donors = data.get("donor_features")
    if not isinstance(donors, list) or not donors:
        warnings.append("donor_features is empty; design may be task-only rather than repository-pattern grounded.")

    contracts = data.get("propagation_contracts")
    if not isinstance(contracts, list) or not contracts:
        errors.append("design_contract.propagation_contracts must contain required architecture roles.")
    else:
        required_count = 0
        roles: set[str] = set()
        for idx, item in enumerate(contracts):
            if not isinstance(item, dict):
                errors.append(f"propagation_contracts[{idx}] must be an object.")
                continue
            cid = item.get("id")
            role = item.get("role")
            if not _nonempty_str(cid):
                errors.append(f"propagation_contracts[{idx}].id must be non-empty.")
            if role not in ROLE_NAMES:
                errors.append(
                    f"propagation_contracts[{idx}].role={role!r} is not allowed. "
                    f"Allowed roles: {sorted(ROLE_NAMES)}"
                )
            else:
                roles.add(str(role))
            if item.get("required", True):
                required_count += 1
            if not _nonempty_str(item.get("constraint")):
                errors.append(f"propagation_contracts[{idx}].constraint must be non-empty.")
            if not isinstance(item.get("failure_if_omitted"), str) or not item.get("failure_if_omitted", "").strip():
                warnings.append(f"propagation_contracts[{idx}] should explain failure_if_omitted.")
            _check_files_field(repo_path, item, f"propagation_contracts[{idx}]", errors, warnings)

        if required_count == 0:
            errors.append("At least one propagation contract must be required.")
        if "core_behavior" not in roles:
            errors.append("Design contract must include a core_behavior role.")
        if not ({"public_api", "binding", "caller_compat", "registration", "serialization"} & roles):
            warnings.append("No propagation role beyond core_behavior was found; design may miss integration closure.")

    text = json.dumps(data, ensure_ascii=False)
    if "FAIL_TO_PASS" in text:
        errors.append("design_contract must not mention hidden benchmark oracle field FAIL_TO_PASS.")
    if "spec_goal" in text:
        warnings.append("design_contract mentions legacy spec_goal; avoid executable generated test goals.")

    return _result("DESIGN CONTRACT CHECK", errors, warnings)


def check_role_coverage(contract_path: Path, coverage_path: Path, repo_path: Path) -> ContractCheck:
    contract, error = _load_json(contract_path)
    if error:
        return ContractCheck(False, error)
    coverage, error = _load_json(coverage_path)
    if error:
        return ContractCheck(False, error)

    changed_files = _git_changed_files(repo_path)
    errors: list[str] = []
    warnings: list[str] = []

    entries = coverage.get("role_coverage")
    if not isinstance(entries, list) or not entries:
        errors.append("role_coverage.json must contain a non-empty role_coverage list.")
        entries = []

    by_id: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"role_coverage[{idx}] must be an object.")
            continue
        cid = entry.get("id")
        if not _nonempty_str(cid):
            errors.append(f"role_coverage[{idx}].id must be non-empty.")
            continue
        by_id[str(cid)] = entry

    for item in contract.get("propagation_contracts", []):
        if not isinstance(item, dict) or not item.get("required", True):
            continue
        cid = str(item.get("id", ""))
        if not cid:
            continue
        entry = by_id.get(cid)
        if entry is None:
            errors.append(f"Missing role coverage entry for required contract {cid}.")
            continue
        status = entry.get("status")
        if status not in {"done", "already_satisfied", "not_applicable"}:
            errors.append(
                f"Coverage for {cid} has invalid status {status!r}; use done, "
                "already_satisfied, or not_applicable."
            )
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"Coverage for {cid} must include non-empty evidence list.")
        if status == "not_applicable" and not _nonempty_str(entry.get("reason")):
            errors.append(f"Coverage for {cid} marked not_applicable must include a reason.")
        contract_files = set(_normalize_files(item.get("files")))
        if status == "done" and contract_files and not (contract_files & changed_files):
            warnings.append(
                f"Coverage for {cid} is done but patch did not touch declared files "
                f"{sorted(contract_files)}. Evidence must justify this."
            )

    anti = contract.get("anti_contracts", [])
    if isinstance(anti, list):
        for item in anti:
            if isinstance(item, dict) and item.get("id") in {
                "no_generated_spec_tests",
                "no_broad_generated_tests",
            }:
                generated = [p for p in changed_files if "spec_goal" in p or p.startswith("tests/generated")]
                if generated:
                    errors.append(f"Patch violates no_generated_spec_tests: {sorted(generated)}")

    if not changed_files:
        errors.append("No repository changes detected; implementation cannot satisfy the task.")

    return _result("ROLE COVERAGE CHECK", errors, warnings)


def write_feedback(path: Path, check: ContractCheck) -> None:
    path.write_text(check.output, encoding="utf-8")


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"{path} does not exist."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"{path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return {}, f"{path} must contain a JSON object."
    return data, None


def _result(title: str, errors: list[str], warnings: list[str]) -> ContractCheck:
    lines = [f"== {title} =="]
    lines.extend(f"ERROR: {msg}" for msg in errors)
    lines.extend(f"WARNING: {msg}" for msg in warnings)
    if not errors:
        lines.append("PASS")
    return ContractCheck(not errors, "\n".join(lines) + "\n")


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_files(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    files: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if not path:
            continue
        if path.startswith("/testbed/"):
            path = path.removeprefix("/testbed/")
        files.append(path)
    return files


def _check_files_field(
    repo_path: Path,
    item: dict[str, Any],
    prefix: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    files = _normalize_files(item.get("files"))
    if "files" not in item:
        errors.append(f"{prefix}.files must be present.")
        return
    if not files:
        warnings.append(f"{prefix}.files is empty; use [] only when role is repository-wide or conceptual.")
    for file in files:
        if not (repo_path / file).exists():
            warnings.append(f"{prefix}.files references missing path {file!r}; verify it is created by the task.")


def _git_changed_files(repo_path: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--name-only"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    files = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    result = subprocess.run(
        ["git", "-C", str(repo_path), "ls-files", "--others", "--exclude-standard"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    files.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return files
