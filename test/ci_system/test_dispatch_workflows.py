import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from pipeline import build_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_RUNNER_PREFIXES = ("b200-", "amd-", "gb200-", "b300-")
SLURM_RUNNER_PREFIXES = (
    "b200-",
    "gb200-",
    "slurm-b200-",
    "slurm-gb200-",
    "slurm-gb300-",
)
PR_TEST_WORKFLOWS = (
    "amd-mi350-tests.yml",
    "amd-mi450-tests.yml",
    "nvidia-b200-tests.yml",
    "pr-test-nvidia-arm.yml",
)


def workflow_dispatch_inputs(name: str) -> dict:
    path = REPO_ROOT / ".github" / "workflows" / name
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    return triggers["workflow_dispatch"]["inputs"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", PR_TEST_WORKFLOWS)
def test_pr_workflow_can_classify_pull_request_files(name: str):
    workflow = load_yaml(REPO_ROOT / ".github" / "workflows" / name)

    assert workflow["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }


def run_slurm_dispatch_script(
    tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    workflow = load_yaml(REPO_ROOT / ".github/workflows/slurm-dispatch.yml")
    step = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Submit and wait for Slurm tasks"
    )
    original_script = step["run"]
    script = original_script.replace(
        'exec test/ci/run_slurm.sh "${args[@]}"',
        """printf 'arg=%s\\n' "${args[@]}"
printf 'artifact=%s\\n' "${TS_CI_ARTIFACT_ROOT-}"
printf 'cache=%s\\n' "${TS_CI_CACHE_DIR-}"
printf 'image=%s\\n' "${TS_CI_CONTAINER_IMAGE-}"
""",
    )
    script = script.replace(
        "repo = Path.cwd()",
        'repo = Path(__import__("os").environ.get("TOKENSPEED_TEST_REPO_ROOT", Path.cwd()))',
    )
    assert script != original_script
    env = {
        **os.environ,
        "PR": "",
        "CONTAINER_IMAGE": "",
        "CLUSTER": "gb200",
        "YAML_SELECTION": "off",
        "RUNNERS": workflow_dispatch_inputs("slurm-dispatch.yml")["runners"]["default"],
        "TASK_TYPES": "eval,perf",
        "MATCH": "",
        "INCLUDE_MMLU": "false",
        "TRIGGER": "all",
        "RUNNER_TEMP": str(tmp_path),
        "USER": "test-coordinator",
        **overrides,
    }
    env.pop("TS_CI_ARTIFACT_ROOT", None)
    env.pop("TS_CI_CACHE_DIR", None)
    env.pop("TS_CI_CONTAINER_IMAGE", None)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_k8s_resolve_source_script(
    tmp_path: Path,
    *,
    pr: str = "",
    commit: str = "",
    pr_files: str = "README.md",
) -> tuple[subprocess.CompletedProcess[str], str, str, str]:
    workflow = load_yaml(REPO_ROOT / ".github/workflows/k8s-dispatch.yml")
    step = next(
        step
        for step in workflow["jobs"]["scan"]["steps"]
        if step.get("name") == "Resolve source"
    )
    script = step["run"].replace("${{ github.repository }}", "lightseekorg/tokenspeed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "gh-calls"
    output = tmp_path / "github-output"
    summary = tmp_path / "step-summary"
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$GH_CALLS"
endpoint=${2:-}
case "$endpoint" in
  repos/lightseekorg/tokenspeed/commits/main)
    printf '%s\\n' "$GH_MAIN_SHA"
    ;;
  repos/lightseekorg/tokenspeed/commits/*)
    printf '{"sha":"%s","html_url":"https://github.com/lightseekorg/tokenspeed/commit/%s"}\\n' \
      "$GH_COMMIT_SHA" "$GH_COMMIT_SHA"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*/files)
    printf '%s\\n' "$GH_PR_FILES"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*)
    printf '{"base":{"sha":"%s"},"head":{"sha":"%s"},"html_url":"https://github.com/lightseekorg/tokenspeed/pull/123"}\\n' \
      "$GH_PR_BASE_SHA" "$GH_PR_SHA"
    ;;
  *)
    echo "Unexpected gh call: $*" >&2
    exit 1
    ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "test-token",
        "GH_CALLS": str(calls),
        "GH_MAIN_SHA": "1" * 40,
        "GH_COMMIT_SHA": commit.strip().lower(),
        "GH_PR_BASE_SHA": "3" * 40,
        "GH_PR_SHA": "2" * 40,
        "GH_PR_FILES": pr_files,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "PR": pr,
        "COMMIT": commit,
    }
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result,
        output.read_text(encoding="utf-8") if output.exists() else "",
        summary.read_text(encoding="utf-8") if summary.exists() else "",
        calls.read_text(encoding="utf-8") if calls.exists() else "",
    )


def run_deepswe_resolve_script(tmp_path: Path, *, pr: str, pr_files: str) -> str:
    workflow = load_yaml(REPO_ROOT / ".github/workflows/b300-deepswe.yml")
    step = next(
        step
        for step in workflow["jobs"]["source"]["steps"]
        if step.get("name") == "Resolve trusted source"
    )
    script = step["run"].replace("${{ github.repository }}", "lightseekorg/tokenspeed")
    for placeholder in ("task_count", "sample_seed", "concurrency"):
        script = script.replace(f"${{{{ inputs.{placeholder!s} }}}}", "1")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${2:-}" in
  repos/lightseekorg/tokenspeed/commits/main)
    printf '%s\\n' "$GH_MAIN_SHA"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*/files)
    printf '%s\\n' "$GH_PR_FILES"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*)
    printf '{"head":{"sha":"%s","repo":{"full_name":"%s"}},"html_url":"%s"}\\n' \
      "$GH_PR_SHA" lightseekorg/tokenspeed "https://example.invalid/pull/7"
    ;;
  *)
    echo "Unexpected gh call: $*" >&2
    exit 1
    ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GH_TOKEN": "test-token",
            "GH_MAIN_SHA": "1" * 40,
            "GH_PR_SHA": "2" * 40,
            "GH_PR_FILES": pr_files,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary"),
            "PR": pr,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output.read_text(encoding="utf-8")


def eligible_config_paths(runner_prefixes: tuple[str, ...]) -> set[str]:
    paths = set()
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = task["runner"]["labels"]
        if any(label.startswith(runner_prefixes) for label in labels):
            paths.add(path.relative_to(REPO_ROOT).as_posix())
    return paths


def eligible_slurm_config_paths() -> set[str]:
    paths = eligible_config_paths(SLURM_RUNNER_PREFIXES)
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = task["runner"]["labels"]
        if task["type"] != "perf" and any(
            label.startswith("b300-") for label in labels
        ):
            paths.add(path.relative_to(REPO_ROOT).as_posix())
    return paths


def configured_yaml_choices(workflow_name: str) -> set[str]:
    choices = workflow_dispatch_inputs(workflow_name)["yaml"]["options"]
    return {choice for choice in choices if choice.startswith("test/ci/")}


def test_k8s_dispatch_lists_every_supported_ci_yaml():
    choices = configured_yaml_choices("k8s-dispatch.yml")
    assert eligible_config_paths(K8S_RUNNER_PREFIXES) <= choices
    assert all((REPO_ROOT / choice).is_file() for choice in choices)


def test_kernel_ut_tasks_cover_every_test_once_per_runner():
    # NVIDIA runners split the kernel suite across part-i and part-ii; AMD
    # runs it in a single task.
    root = REPO_ROOT / "tokenspeed-kernel/test"
    files = {
        path
        for path in root.rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }
    assert files
    tasks = [
        load_yaml(REPO_ROOT / f"test/ci/ut/ut-tokenspeed-kernel{suffix}.yaml")
        for suffix in ("-part-i", "-part-ii", "")
    ]
    labels = {label for task in tasks for label in task["runner"]["labels"]}
    # Pin the runner set so dropping a runner from every task fails here.
    assert labels == {
        "h100-1gpu",
        "b200-1gpu",
        "gb200-1gpu",
        "b300-1gpu",
        "amd-mi355-1gpu",
    }
    for label in labels:
        executions = Counter()
        for task in tasks:
            if label not in task["runner"]["labels"]:
                continue
            for command in task["ut"]["commands"]:
                tokens = shlex.split(command)
                arguments = tokens[tokens.index("pytest") + 1 :]
                targets = [
                    REPO_ROOT / arg for arg in arguments if not arg.startswith("-")
                ]
                ignores = [
                    REPO_ROOT / arg.removeprefix("--ignore=")
                    for arg in arguments
                    if arg.startswith("--ignore=")
                ]
                executions.update(
                    path
                    for path in files
                    if any(path.is_relative_to(target) for target in targets)
                    and not any(path.is_relative_to(ignore) for ignore in ignores)
                )

        assert executions == Counter({path: 1 for path in files}), label


def test_runtime_1gpu_tasks_run_every_partition_once_per_runner():
    # AMD splits the runtime-1gpu suite across part-i and part-ii; other
    # runners run it whole in ut-runtime-1gpu.
    partitions = {}
    for suffix in ("", "-part-i", "-part-ii"):
        task = load_yaml(REPO_ROOT / f"test/ci/ut/ut-runtime-1gpu{suffix}.yaml")
        for command in task["ut"]["commands"]:
            tokens = shlex.split(command)
            if "run_ci_suite.py" not in command:
                continue
            partition = (0, 1)
            if "--auto-partition-size" in tokens:
                partition = (
                    int(tokens[tokens.index("--auto-partition-id") + 1]),
                    int(tokens[tokens.index("--auto-partition-size") + 1]),
                )
            for label in task["runner"]["labels"]:
                partitions.setdefault(label, []).append(partition)

    # Pin the runner set so dropping a runner from every task fails here.
    assert partitions.keys() == {
        "h100-1gpu",
        "b200-1gpu",
        "gb200-1gpu",
        "amd-mi355-1gpu",
    }
    for label, parts in partitions.items():
        size = parts[0][1]
        assert sorted(parts) == [(i, size) for i in range(size)], label


def test_amd_kernel_benchmarks_run_in_their_own_workflow():
    # AMD Tests keeps unit and model tests; the benchmark lives in a separate
    # workflow so it starts alongside unit tests and completes on its own.
    tests = load_yaml(REPO_ROOT / ".github/workflows/amd-mi350-tests.yml")["jobs"]
    assert "kernel-benchmark" not in tests
    assert "kernel-benchmark" not in tests["finish"]["needs"]
    assert (
        "kernel-benchmark"
        not in next(
            step
            for step in tests["scan"]["steps"]
            if step.get("name") == "Build task matrix"
        )["run"]
    )

    workflow = load_yaml(REPO_ROOT / ".github/workflows/amd-kernel-benchmark.yml")
    assert workflow["name"] == "AMD Kernel Benchmark"
    triggers = workflow.get("on") or workflow.get(True)
    # The `high priority` label only affects model tests; it must not restart
    # the benchmark.
    assert "labeled" not in triggers["pull_request"]["types"]
    # tokenspeed-mla is NVIDIA-only, so its changes must not start the workflow.
    for event in ("push", "pull_request"):
        assert "tokenspeed-mla/**" not in triggers[event]["paths"]
    jobs = workflow["jobs"]
    assert (
        '--workflow-stage "kernel-benchmark"'
        in next(
            step
            for step in jobs["scan"]["steps"]
            if step.get("name") == "Build task matrix"
        )["run"]
    )
    benchmark = jobs["kernel-benchmark"]
    assert benchmark["needs"] == "scan"
    assert "draft == false" in benchmark["if"]
    assert (
        "github.event.pull_request.base.sha" in benchmark["with"]["comparison_base_ref"]
    )
    assert (
        "github.event.pull_request.head.sha"
        in benchmark["with"]["comparison_candidate_ref"]
    )
    assert "always()" in jobs["finish"]["if"]


@pytest.mark.parametrize(
    ("scan", "benchmark", "passes"),
    [
        ("success", "success", True),
        ("success", "skipped", True),
        ("success", "failure", False),
        ("success", "cancelled", False),
        # A failed scan skips the benchmark and must not pass the gate.
        ("failure", "skipped", False),
        ("cancelled", "skipped", False),
    ],
)
def test_amd_kernel_benchmark_gate_blocks_on_failure(scan, benchmark, passes):
    workflow = load_yaml(REPO_ROOT / ".github/workflows/amd-kernel-benchmark.yml")
    # The default-branch ruleset requires a check named `finish`; a renamed
    # gate job would stop blocking merges.
    finish = workflow["jobs"]["finish"]
    assert "name" not in finish
    assert finish["needs"] == ["scan", "kernel-benchmark"]
    (step,) = finish["steps"]
    completed = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "SCAN_RESULT": scan,
            "KERNEL_BENCHMARK_RESULT": benchmark,
        },
        capture_output=True,
        text=True,
    )
    assert (completed.returncode == 0) == passes


def test_kernel_benchmark_task_uses_shared_ci_contract():
    task = load_yaml(REPO_ROOT / "test/ci/perf/kernel-benchmark-amd-gfx950.yaml")

    assert task["type"] == "perf"
    assert task["workflow_stage"] == "kernel-benchmark"
    assert task["triggers"] == ["per-commit", "manual"]
    assert task["runner"]["labels"] == ["amd-mi355-1gpu"]
    assert task["env"]["TOKENSPEED_KERNEL_BENCHMARK_PROFILER"] == "none"
    assert ".ci-artifacts/published" in task["perf"]["command"]
    for variable in ("BASE_REF", "CANDIDATE_REF", "PR_NUMBER", "MERGE_SHA"):
        assert variable in task["perf"]["command"]


def test_k8s_dispatch_accepts_full_commit_sha(tmp_path):
    requested = "A" * 40

    result, output, summary, calls = run_k8s_resolve_source_script(
        tmp_path, commit=f"  {requested}  "
    )

    assert result.returncode == 0, result.stderr
    assert f"sha={'a' * 40}" in output
    assert f"comparison_base_ref={'1' * 40}" in output
    assert f"comparison_candidate_ref={'a' * 40}" in output
    assert "install_mla=1" in output
    assert "- Mode: commit" in summary
    assert f"api repos/lightseekorg/tokenspeed/commits/{'a' * 40}" in calls


def test_k8s_dispatch_defaults_to_latest_main(tmp_path):
    result, output, summary, calls = run_k8s_resolve_source_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert f"sha={'1' * 40}" in output
    assert f"comparison_base_ref={'1' * 40}" in output
    assert f"comparison_candidate_ref={'1' * 40}" in output
    assert "install_mla=0" in output
    assert "- Mode: main" in summary
    assert "api repos/lightseekorg/tokenspeed/commits/main --jq .sha" in calls


@pytest.mark.parametrize(
    ("pr_files", "expected_install_mla"),
    [("README.md", "0"), ("tokenspeed-mla/src/kernel.py", "1")],
)
def test_k8s_dispatch_preserves_pr_resolution(tmp_path, pr_files, expected_install_mla):
    result, output, summary, calls = run_k8s_resolve_source_script(
        tmp_path,
        pr=" https://github.com/lightseekorg/tokenspeed/pull/123 ",
        pr_files=pr_files,
    )

    assert result.returncode == 0, result.stderr
    assert f"sha={'2' * 40}" in output
    assert f"comparison_base_ref={'3' * 40}" in output
    assert f"comparison_candidate_ref={'2' * 40}" in output
    assert f"install_mla={expected_install_mla}" in output
    assert "- Mode: pr" in summary
    assert "api repos/lightseekorg/tokenspeed/pulls/123" in calls


def test_k8s_dispatch_rejects_pr_and_commit_together(tmp_path):
    result, output, _, calls = run_k8s_resolve_source_script(
        tmp_path, pr="123", commit="a" * 40
    )

    assert result.returncode == 2
    assert "Set only one of pr or commit" in result.stderr
    assert output == ""
    assert calls == ""


@pytest.mark.parametrize("commit", ["a" * 39, "feature-branch", "a" * 41])
def test_k8s_dispatch_rejects_non_full_commit_sha(tmp_path, commit):
    result, output, _, calls = run_k8s_resolve_source_script(tmp_path, commit=commit)

    assert result.returncode == 2
    assert "expected exactly 40 hexadecimal characters" in result.stderr
    assert output == ""
    assert calls == ""


def test_k8s_dispatch_commit_input_is_optional():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/k8s-dispatch.yml")
    commit_input = workflow_dispatch_inputs("k8s-dispatch.yml")["commit"]

    assert commit_input["required"] is False
    assert commit_input["default"] == ""
    assert "full commit SHA" in commit_input["description"]
    assert (
        "${{ inputs.commit || inputs.pr || 'main' }}"
        in workflow["concurrency"]["group"]
    )


def test_k8s_dispatch_passes_comparison_revisions_to_tasks():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/k8s-dispatch.yml")
    run_inputs = workflow["jobs"]["run"]["with"]

    assert run_inputs["comparison_base_ref"] == (
        "${{ needs.scan.outputs.comparison_base_ref }}"
    )
    assert run_inputs["comparison_candidate_ref"] == (
        "${{ needs.scan.outputs.comparison_candidate_ref }}"
    )


def test_slurm_dispatch_lists_every_supported_ci_yaml():
    assert (
        configured_yaml_choices("slurm-dispatch.yml") == eligible_slurm_config_paths()
    )


def test_slurm_dispatch_lists_every_supported_trigger():
    choices = workflow_dispatch_inputs("slurm-dispatch.yml")["trigger"]["options"]
    assert set(choices) == {
        "all",
        "per-commit",
        "manual",
        "nightly",
        "debug",
        "slurm",
    }


def test_slurm_dispatch_lists_every_supported_cluster():
    cluster = workflow_dispatch_inputs("slurm-dispatch.yml")["cluster"]

    assert cluster["default"] == "gb200"
    assert set(cluster["options"]) == {"gb200", "gb300"}


def test_slurm_dispatch_accepts_immutable_tokenspeed_image_override(tmp_path):
    image = (
        "ghcr.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64
    )

    result = run_slurm_dispatch_script(tmp_path, CONTAINER_IMAGE=image)

    assert result.returncode == 0, result.stderr
    assert f"image={image}" in result.stdout


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18",
        "ghcr.io/other/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64,
        "docker.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64,
    ],
)
def test_slurm_dispatch_rejects_unsafe_image_override(tmp_path, image):
    result = run_slurm_dispatch_script(tmp_path, CONTAINER_IMAGE=image)

    assert result.returncode == 2
    assert "container_image must be an immutable" in result.stderr


def test_slurm_dispatch_routes_gb300_to_its_coordinator():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/slurm-dispatch.yml")
    checkout = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Checkout trusted dispatcher"
    )
    dispatch_script = next(
        step["run"]
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Submit and wait for Slurm tasks"
    )

    assert workflow["jobs"]["dispatch"]["runs-on"] == (
        "${{ inputs.cluster == 'gb300' && "
        "'slurm-dispatch-gb300' || 'slurm-dispatch' }}"
    )
    assert "${{ inputs.cluster }}" in workflow["concurrency"]["group"]
    assert checkout["with"]["ref"] == "${{ inputs.pr && 'main' || github.sha }}"
    assert "${{ github.ref }}" in workflow["concurrency"]["group"]
    assert 'python3 - "$YAML_SELECTION" "$CLUSTER" "$PR"' in dispatch_script
    assert "from slurm_submit import pr_worktree" in dispatch_script
    assert "with pr_worktree(repo, pr) as checkout:" in dispatch_script


@pytest.mark.parametrize("runners", ["b200-4gpu,gb200-4gpu", "b200-4gpu, gb200-4gpu"])
def test_slurm_dispatch_uses_declared_gb300_runner_and_shared_paths(tmp_path, runners):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS=runners,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )
    assert "arg=b200-4gpu" not in result.stdout
    assert "arg=gb200-4gpu" not in result.stdout
    assert "artifact=/data/home/test-coordinator/tokenspeed-slurm" in result.stdout
    assert "cache=/data/home/test-coordinator/tokenspeed-cache" in result.stdout


def test_slurm_dispatch_preserves_gb200_defaults(tmp_path):
    result = run_slurm_dispatch_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "arg=--runner\narg=b200-4gpu\n" in result.stdout
    assert "arg=--runner\narg=gb200-4gpu\n" in result.stdout
    assert "arg=--runner\narg=slurm-gb200-4gpu\n" in result.stdout
    assert "artifact=\n" in result.stdout
    assert "cache=\n" in result.stdout
    assert "image=\n" in result.stdout


def test_slurm_dispatch_maps_gb300_defaults_without_changing_filters(tmp_path):
    result = run_slurm_dispatch_script(tmp_path, CLUSTER="gb300")

    assert result.returncode == 0, result.stderr
    assert "arg=--all\n" in result.stdout
    assert "arg=--runner-alias\narg=b200-4gpu=gb300-4gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb200-4gpu=gb300-4gpu\n" in result.stdout
    assert (
        "arg=--runner-alias\narg=slurm-gb200-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )
    assert "arg=--type\narg=eval\n" in result.stdout
    assert "arg=--type\narg=perf\n" in result.stdout
    assert "arg=--exclude-match\narg=mmlu\n" in result.stdout
    assert "artifact=/data/home/test-coordinator/tokenspeed-slurm" in result.stdout
    assert "cache=/data/home/test-coordinator/tokenspeed-cache" in result.stdout


def test_slurm_dispatch_resolves_missing_coordinator_user(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        USER="",
    )
    coordinator_user = subprocess.run(
        ["id", "-un"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert result.returncode == 0, result.stderr
    assert f"artifact=/data/home/{coordinator_user}/tokenspeed-slurm" in result.stdout
    assert f"cache=/data/home/{coordinator_user}/tokenspeed-cache" in result.stdout


def test_slurm_dispatch_rejects_runner_for_another_cluster(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS="b200-4gpu",
    )

    assert result.returncode == 2
    assert "does not identify exactly one runner declared" in result.stderr


def test_slurm_dispatch_maps_b200_yaml_to_gb300_runners(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel-part-i.yaml",
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=b200-1gpu=gb300-1gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb200-1gpu=gb300-1gpu\n" in result.stdout


def test_slurm_dispatch_accepts_one_explicit_matching_gb300_runner(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS="slurm-gb300-4gpu",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )


def test_slurm_dispatch_passes_multi_node_gb300_runner_unchanged(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )


@pytest.mark.parametrize(
    ("runners", "message"),
    [
        ("gb300-4gpu", "does not identify exactly one runner declared"),
        ("slurm-gb300-4gpu,slurm-gb300-4gpu", "more than once"),
    ],
)
def test_slurm_dispatch_rejects_mismatched_or_multiple_gb300_runners(
    tmp_path, runners, message
):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS=runners,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_slurm_dispatch_accepts_multiple_native_gb300_runners(tmp_path):
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-part-i.yaml")
    task["runner"]["labels"] = ["gb300-1gpu", "gb300-4gpu"]
    config = tmp_path / "ambiguous.yaml"
    config.write_text(yaml.safe_dump(task))

    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=str(config),
        TOKENSPEED_TEST_REPO_ROOT=str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=gb300-1gpu=gb300-1gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb300-4gpu=gb300-4gpu\n" in result.stdout


def test_only_dedicated_tasks_declare_gb300():
    configs = []
    for path in (REPO_ROOT / "test/ci").rglob("*.yaml"):
        if any("gb300-" in label for label in load_yaml(path)["runner"]["labels"]):
            configs.append(path.name)

    assert sorted(configs) == [
        "deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml",
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml",
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml",
        "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "qwen3.8-flash-next-nvfp4-tp4ep4-evalscope-agentic.yaml",
    ]


def test_kimi_k3_gb300_is_two_node_nightly_tp8():
    task = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )

    assert task["triggers"] == ["nightly"]
    assert task["runner"]["labels"] == ["slurm-gb300-4gpu"]
    assert task["slurm"] == {"nodes": 2, "gpus_per_node": 4}
    assert "--tensor-parallel-size 8" in task["server"]["command"]


def test_kimi_k3_nvfp4_gb300_uses_pinned_local_models():
    target_path = (
        "/models/nvidia--Kimi-K3-NVFP4/" "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
    )
    draft_path = (
        "/models/Inferact--Kimi-K3-DSpark/" "cf6b8244620e7ea4b0651d214f28e89eac75bed6"
    )
    plain = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )
    dspark = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )

    assert target_path in plain["server"]["command"]
    assert target_path in dspark["server"]["command"]
    assert draft_path in dspark["server"]["command"]
    assert dspark["env"]["TOKENSPEED_DFLASH_AUX_STREAM"] == "attn_res"


def test_kimi_k3_dflash2_gb300_uses_a_window_aware_drafter_backend():
    task = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )
    command = task["server"]["command"]

    assert task["slurm"] == {"nodes": 2, "gpus_per_node": 4}
    assert "/models/nvidia--Kimi-K3-NVFP4/" in command
    assert "--speculative-draft-model-path lightseekorg/kimi-k3-dflash2" in command
    assert "--speculative-algorithm DFLASH" in command
    # The draft's sliding_attention layers need a backend that masks with them.
    assert "--drafter-attention-backend mla" in command


def test_gb300_slurm_nightly_workflow_is_scheduled_and_isolated():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/nvidia-gb300-nightly-tests.yml")
    triggers = workflow.get("on") or workflow.get(True)
    scan = workflow["jobs"]["scan"]
    submit = workflow["jobs"]["submit"]
    gate = next(
        step for step in scan["steps"] if step.get("name") == "Check trusted source"
    )
    matrix_step = next(
        step
        for step in scan["steps"]
        if step.get("name") == "Build nightly GB300 task matrix"
    )
    submit_script = next(
        step["run"]
        for step in submit["steps"]
        if step.get("name") == "Submit and wait for nightly GB300 Slurm task"
    )
    checkout = next(
        step for step in submit["steps"] if step.get("name") == "Checkout dispatcher"
    )

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "17 18 * * *"}]
    assert "github.repository == 'lightseekorg/tokenspeed'" in gate["env"]["ALLOWED"]
    assert "vars.TOKENSPEED_CI_REPOSITORY" in gate["env"]["ALLOWED"]
    assert "github.ref == 'refs/heads/main'" in gate["env"]["ALLOWED"]
    assert gate["env"]["ENABLED"] == (
        "${{ vars.TOKENSPEED_CI_GB300_SLURM_NIGHTLY_ENABLED == 'true' }}"
    )
    gate_condition = (
        "steps.gate.outputs.allowed == 'true' && "
        "steps.gate.outputs.enabled == 'true'"
    )
    for step_name in (
        "Checkout code",
        "Install scan dependency",
        "Build nightly GB300 task matrix",
    ):
        step = next(step for step in scan["steps"] if step.get("name") == step_name)
        assert step["if"] == gate_condition
    assert workflow["concurrency"] == {
        "group": "gb300-slurm-nightly",
        "cancel-in-progress": False,
    }
    assert submit["name"] == "${{ matrix.name }}"
    assert submit["runs-on"] == "slurm-dispatch-gb300"
    assert "needs.scan.outputs.allowed == 'true'" in submit["if"]
    assert "needs.scan.outputs.enabled == 'true'" in submit["if"]
    assert "needs.scan.outputs.has_tasks == 'true'" in submit["if"]
    assert "--trigger nightly" in matrix_step["run"]
    assert "--runner-group nvidia-arm" in matrix_step["run"]
    assert "--workflow-stage model-test" in matrix_step["run"]
    assert "--multi-node only" in matrix_step["run"]
    assert "startswith('slurm-gb300-')" in matrix_step["run"]
    assert '--runner "$RUNNER"' in submit_script
    assert "--source-pr" not in submit_script
    assert "secrets.HF_TOKEN" not in str(submit)
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN" in submit_script
    scan_checkout = next(
        step for step in scan["steps"] if step.get("name") == "Checkout code"
    )
    assert scan_checkout["with"]["ref"] == "${{ github.sha }}"
    assert scan_checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False


def test_gb300_slurm_nightly_matrix_selects_all_nightly_kimi_k3_tasks(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", raising=False)

    matrix = build_matrix(
        REPO_ROOT / "test/ci",
        REPO_ROOT,
        trigger="nightly",
        runner_group="nvidia-arm",
        workflow_stage="model-test",
        multi_node="only",
    )

    assert {(entry["config"], entry["runner"]) for entry in matrix["include"]} == {
        (
            "test/ci/eval/"
            "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-"
            "gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
    }


def test_gb300_slurm_per_commit_workflow_is_isolated_and_automatic():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/nvidia-gb300-tests.yml")
    triggers = workflow.get("on") or workflow.get(True)
    submit = workflow["jobs"]["submit"]
    scan_steps = workflow["jobs"]["scan"]["steps"]
    gate = next(
        step for step in scan_steps if step.get("name") == "Check trusted source"
    )
    matrix_step = next(
        step
        for step in scan_steps
        if step.get("name") == "Build multi-node task matrix"
    )
    submit_script = next(
        step["run"]
        for step in submit["steps"]
        if step.get("name") == "Submit and wait for GB300 Slurm task"
    )
    checkout = next(
        step for step in submit["steps"] if step.get("name") == "Checkout dispatcher"
    )

    assert set(triggers) == {"push", "pull_request"}
    assert submit["name"] == "${{ matrix.name }}"
    assert submit["runs-on"] == "slurm-dispatch-gb300"
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert '--runner "$RUNNER"' in submit_script
    assert "--runner-alias" not in submit_script
    assert '--source-pr "$PR_NUMBER"' in submit_script
    assert '--pr "$PR_NUMBER"' not in submit_script
    assert "secrets.HF_TOKEN" not in str(submit)
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN" in submit_script
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    assert "github.repository == 'lightseekorg/tokenspeed'" in gate["env"]["ALLOWED"]
    assert "github.event.pull_request.draft == false" in gate["env"]["ALLOWED"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in gate["env"]["ALLOWED"]
    )
    assert gate["env"]["ENABLED"] == (
        "${{ vars.TOKENSPEED_CI_GB300_SLURM_PER_COMMIT_ENABLED == 'true' }}"
    )
    assert "needs.scan.outputs.enabled == 'true'" in submit["if"]
    assert "TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS" not in matrix_step.get("env", {})
    assert "--multi-node only" in matrix_step["run"]

    cancel_workflow = load_yaml(
        REPO_ROOT / ".github/workflows/cancel-pr-tests-on-close.yml"
    )
    cancel_groups = {
        item["group"]
        for item in cancel_workflow["jobs"]["cancel"]["strategy"]["matrix"]["include"]
    }
    assert "gb300-slurm-per-commit" in cancel_groups


def test_gb300_slurm_per_commit_matrix_excludes_nightly_tp8_tasks(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", raising=False)

    matrix = build_matrix(
        REPO_ROOT / "test/ci",
        REPO_ROOT,
        trigger="per-commit",
        runner_group="nvidia-arm",
        workflow_stage="model-test",
        multi_node="only",
    )

    assert matrix["include"] == [
        {
            "name": "eval-deepseek-v4.1-flash-pd-1p1d-dspark-gsm8k-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
        {
            "name": "eval-kimi-k3-nvfp4-dp16-four-node-aime26-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
        {
            "name": "eval-kimi-k3-nvfp4-dspark-tp8-two-node-aime26-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
    ]


def test_nvidia_arm_workflow_excludes_multi_node_tasks():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")
    scan_script = next(
        step["run"]
        for step in workflow["jobs"]["scan"]["steps"]
        if step.get("name") == "Build task matrix"
    )

    assert "--multi-node exclude" in scan_script


def test_qwen35_agentic_allows_declared_80k_context():
    task = load_yaml(
        REPO_ROOT / "test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml"
    )

    assert "--max-model-len 80000" in task["server"]["command"]
    assert task["env"]["TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_nvidia_arm_model_tests_allow_runner_wait_time():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")

    assert workflow["jobs"]["model-test"]["with"]["timeout_minutes"] >= 120


def test_mi450_sim_uses_direct_runner_and_bounded_timeout():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/run-ci-task-matrix.yml")
    job = workflow["jobs"]["test"]

    assert job["runs-on"] == "${{ matrix.runner }}"
    assert job["timeout-minutes"] == (
        "${{ matrix.runner == 'amd-mi45x-cpu-test'"
        " && 30 || inputs.timeout_minutes }}"
    )


@pytest.mark.parametrize(
    ("workflow_stage", "task_type"),
    [
        ("unit-test", "ut"),
        ("kernel-benchmark", "perf"),
        ("model-test", "eval"),
        ("model-test", "perf"),
    ],
)
def test_pr_task_caches_are_isolated_and_cleaned_with_their_job(
    tmp_path, workflow_stage, task_type
):
    workflow = load_yaml(REPO_ROOT / ".github/workflows/run-ci-task-matrix.yml")
    steps = workflow["jobs"]["test"]["steps"]
    setup = next(step for step in steps if step["name"] == "Set work directory")
    cleanup = next(step for step in steps if step["name"] == "Cleanup work directory")
    assert cleanup["if"] == "always()"
    shared_cache = tmp_path / "shared-uv"
    shared_cache.mkdir()
    persistent_hf_home = tmp_path / "shared-huggingface"
    persistent_hf_home.mkdir()
    sentinel = shared_cache / "another-job"
    sentinel.touch()
    cache_variables = (
        "UV_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
    )
    isolated_variables = tuple(
        variable
        for variable in cache_variables
        if variable != "TRITON_CACHE_DIR" or task_type == "eval"
    )
    job_envs = []
    for attempt in (1, 2):
        env_file = tmp_path / f"env-{attempt}"
        script = setup["run"]
        for expression, value in {
            "github.workspace": str(tmp_path / "workspace with spaces"),
            "github.run_id": "1234",
            "github.run_attempt": str(attempt),
            "matrix.name": "eval-cache-test",
            "matrix.runner": "model-runner",
            "matrix.workflow_stage": workflow_stage,
            "matrix.type": task_type,
        }.items():
            script = script.replace("${{ " + expression + " }}", value)
        subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "GITHUB_ENV": str(env_file),
                "HF_HOME": str(persistent_hf_home),
                **{variable: str(shared_cache) for variable in cache_variables},
            },
            check=True,
        )
        job_env = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
        assert "MIOPEN_SYSTEM_DB_PATH" not in job_env
        assert "MIOPEN_FIND_MODE" not in job_env
        assert "MIOPEN_FIND_ENFORCE" not in job_env
        if workflow_stage != "model-test":
            assert all(variable not in job_env for variable in cache_variables)
            assert "EVALSCOPE_UV_CACHE_DIR" not in job_env
            continue
        assert job_env["EVALSCOPE_UV_CACHE_DIR"] == str(
            persistent_hf_home / ".uv-cache" / "evalscope"
        )
        if task_type == "perf":
            assert "TRITON_CACHE_DIR" not in job_env
        for variable in isolated_variables:
            cache = Path(job_env[variable])
            assert cache.is_relative_to(Path(job_env["WORK_DIR"]))
            assert cache != shared_cache
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "download").touch()
        job_envs.append(job_env)

    if workflow_stage != "model-test":
        assert sentinel.exists()
        return
    first, second = job_envs
    assert all(first[variable] != second[variable] for variable in isolated_variables)
    script = cleanup["run"].replace("${{ env.WORK_DIR }}", first["WORK_DIR"])
    script = script.replace("${{ matrix.runner }}", "model-runner")
    subprocess.run(["bash", "-c", script], check=True)
    for variable in isolated_variables:
        assert not Path(first[variable]).exists()
        assert (Path(second[variable]) / "download").exists()
    assert sentinel.exists()


def test_gb300_per_commit_forwards_the_tokenspeed_mla_override():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/nvidia-gb300-tests.yml")
    step = next(
        step
        for step in workflow["jobs"]["submit"]["steps"]
        if step.get("name") == "Submit and wait for GB300 Slurm task"
    )

    assert workflow["jobs"]["scan"]["outputs"][
        "install_tokenspeed_mla_from_source"
    ] == ("${{ steps.changes.outputs.install_tokenspeed_mla_from_source }}")
    assert step["env"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == (
        "${{ needs.scan.outputs.install_tokenspeed_mla_from_source }}"
    )


def test_slurm_dispatch_takes_a_dispatched_pr_from_its_own_tree():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/slurm-dispatch.yml")
    step = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Submit and wait for Slurm tasks"
    )

    assert step["env"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == (
        "${{ (inputs.pr || inputs.commit) && '1' || '0' }}"
    )


@pytest.mark.parametrize(
    ("pr", "pr_files", "expected"),
    [
        ("", "", "install_mla=0"),
        ("7", "README.md", "install_mla=0"),
        ("7", "tokenspeed-mla/python/tokenspeed_mla/mla_decode.py", "install_mla=1"),
    ],
)
def test_b300_deepswe_resolves_the_tokenspeed_mla_source(
    tmp_path, pr, pr_files, expected
):
    workflow = load_yaml(REPO_ROOT / ".github/workflows/b300-deepswe.yml")
    assert workflow["jobs"]["run"]["env"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == (
        "${{ needs.source.outputs.install_mla }}"
    )

    assert expected in run_deepswe_resolve_script(tmp_path, pr=pr, pr_files=pr_files)


def test_mi450_sim_runs_on_the_cpu_only_pool():
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml")

    assert task["runner"]["labels"] == ["amd-mi45x-cpu-test"]


def test_mi450_sim_uses_bounded_smoke_suite():
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml")
    setup_script = (REPO_ROOT / "test/ci_system/setup_mi450_sim.sh").read_text()
    parallel_script = (
        REPO_ROOT / "test/ci_system/run_mi450_rocjitsu_parallel.sh"
    ).read_text()

    assert task["env"]["MI450_SIM_RUN_TIMEOUT"] == "600"
    assert task["env"]["MI450_SIM_THREADS_PER_WORKER"] == "2"
    assert 'config["cpu_thread_budget"] = thread_budget' in setup_script
    assert 'threads_per_emulator="${MI450_SIM_THREADS_PER_WORKER:-2}"' in (
        parallel_script
    )
    assert "/sys/fs/cgroup/cpu.max" in parallel_script
    assert task["env"]["MI450_SIM_TEST_ROOT"] != "tokenspeed-kernel/test"
    assert "tokenspeed-kernel/test/amd/ops/attention" in task["env"]["MI450_SIM_TESTS"]


def test_mi450_sim_pins_one_rocm_nightly():
    env = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml")["env"]
    setup_script = (REPO_ROOT / "test/ci_system/setup_mi450_sim.sh").read_text()

    # PyTorch requires the exact ROCm SDK build it was built against.
    rocm_build = f"+rocm{env['ROCM_SDK_VERSION']}"
    for name in ("TORCH_VERSION", "TORCHVISION_VERSION", "TORCH_DEVICE_PACKAGE"):
        assert env[name].endswith(rocm_build), name
    # Standalone runs of the setup script default to the lane's pins, and a
    # cached rocJITsu build is reused only for those same pins.
    for name in ("ROCM_SYSTEMS_REF", "ROCM_SDK_VERSION"):
        assert f"{name}=${{{name}:-{env[name]}}}" in setup_script
    assert 'build_key="${ROCM_SYSTEMS_REF}:${ROCM_SDK_VERSION}"' in setup_script


def test_mi450_sim_uses_stock_triton_compatible_libhip_path():
    script = (REPO_ROOT / "test/ci_system/run_mi450_rocjitsu.sh").read_text()

    assert 'libhip_path="${rocm_root}/lib/libamdhip64.so"' in script
    assert 'test -f "${libhip_path}"' in script
    assert 'export TRITON_LIBHIP_PATH="${libhip_path}"' in script


def npu_step_script(workflow, job, name):
    data = yaml.safe_load((REPO_ROOT / ".github/workflows" / workflow).read_text())
    return next(s["run"] for s in data["jobs"][job]["steps"] if s.get("name") == name)


@pytest.fixture
def npu_source(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    real_git = shutil.which("git")

    def git(*args):
        return subprocess.check_output(
            [real_git, "-C", str(repo), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "CI fixture")
    git("config", "user.email", "fixture@example.invalid")
    for name in ("ci/npu_ops_test/changes.py", ".github/scripts/ci_result_source.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, path)
    git("add", ".")
    git("commit", "--quiet", "-m", "workflow reporter")
    workflow = git("rev-parse", "HEAD")
    reporter = repo / ".github/scripts/ci_result_source.py"
    reporter.write_text(
        reporter.read_text().replace(
            "    actual = subprocess.run(",
            "    raise RuntimeError('candidate reporter must not run')\n    actual = subprocess.run(",
        )
    )
    git("add", ".")
    git("commit", "--quiet", "-m", "candidate source")
    candidate = git("rev-parse", "HEAD")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "fetch-calls"
    waits = tmp_path / "waits"
    wrapper = bin_dir / "git"
    wrapper.write_text(f"#!{sys.executable}\n" + """
import os, pathlib, sys
args = sys.argv[1:]
if 'fetch' in args:
    calls = pathlib.Path(os.environ['FETCH_CALLS'])
    count = int(calls.read_text()) + 1 if calls.exists() else 1
    calls.write_text(str(count))
    if count <= int(os.environ.get('FETCH_FAILURES', '0')):
        print(f'network fixture failure {count}', file=sys.stderr)
        sys.exit(int(os.environ.get('FETCH_STATUS', '128')))
os.execv(os.environ['REAL_GIT'], [os.environ['REAL_GIT'], *args])
""")
    wrapper.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text(f"#!{sys.executable}\n" + """
import os, pathlib, sys, time
with pathlib.Path(os.environ['WAIT_CALLS']).open('a') as out:
    out.write(sys.argv[1] + '\\n')
if os.environ.get('REAL_RETRY_WAIT') == '1':
    time.sleep(float(sys.argv[1]))
""")
    sleep.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REAL_GIT": real_git,
        "FETCH_CALLS": str(calls),
        "WAIT_CALLS": str(waits),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "fixture/tokenspeed",
        "NPU_SOURCE_SHA": candidate,
        "NPU_SOURCE_DIR": str(tmp_path / "checkout"),
        "NPU_WORKFLOW_SHA": candidate,
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{repo.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/fixture/tokenspeed.git",
    }
    return env, calls, waits, workflow, candidate


@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize(
    "failures,status", [(0, 128), (3, 128), (4, 128), (1, 130), (1, 143)]
)
def test_checkout_retry_exhaustion_and_cancellation(
    npu_source, hardware, failures, status, tmp_path
):
    env, calls, waits, workflow, candidate = npu_source
    env.update(FETCH_FAILURES=str(failures), FETCH_STATUS=str(status))
    if hardware:
        env["NPU_REPORTER_DIR"] = str(tmp_path / "task")
    script = npu_step_script(
        "run-ci-task-matrix.yml" if hardware else "pr-test-ascendnpu-arm.yml",
        "test" if hardware else "select",
        "Prepare NPU source and reporter" if hardware else "Check out event commit",
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    exhausted = failures == 4
    cancelled = status in (130, 143)
    assert result.returncode == (status if exhausted or cancelled else 0), result.stderr
    assert int(calls.read_text()) == (4 if failures >= 3 else 1)
    actual_waits = waits.read_text().splitlines() if waits.exists() else []
    assert actual_waits == (["30"] * (3 if failures >= 3 else 0))
    if exhausted or cancelled:
        assert not Path(env["GITHUB_ENV"]).exists()
        assert "network fixture failure" in result.stderr
    else:
        assert (
            subprocess.check_output(
                [env["REAL_GIT"], "-C", env["NPU_SOURCE_DIR"], "rev-parse", "HEAD"],
                text=True,
            ).strip()
            == candidate
        )
        assert (
            subprocess.check_output(
                [env["REAL_GIT"], "-C", env["NPU_SOURCE_DIR"], "status", "--porcelain"],
                text=True,
            )
            == ""
        )
        if hardware:
            assert "NPU_SOURCE_REPORTER=" in Path(env["GITHUB_ENV"]).read_text()


def test_missing_workflow_commit_is_fetched_before_tests(npu_source, tmp_path):
    env, calls, waits, workflow, candidate = npu_source
    env.update(NPU_REPORTER_DIR=str(tmp_path / "task"), NPU_WORKFLOW_SHA=workflow)
    script = npu_step_script(
        "run-ci-task-matrix.yml", "test", "Prepare NPU source and reporter"
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert int(calls.read_text()) == 2
    assert (
        "candidate reporter must not run"
        not in (tmp_path / "task/source-reporter.py").read_text()
    )
    assert (
        subprocess.check_output(
            [env["REAL_GIT"], "-C", env["NPU_SOURCE_DIR"], "rev-parse", "HEAD"],
            text=True,
        ).strip()
        == candidate
    )


def test_reporter_is_offline_pinned_and_rejects_tampering(npu_source, tmp_path):
    env, calls, waits, workflow, candidate = npu_source
    repo = Path(env["NPU_SOURCE_DIR"])
    subprocess.run(
        [env["REAL_GIT"], "clone", "--quiet", str(tmp_path / "repository"), str(repo)],
        check=True,
    )
    subprocess.run(
        [
            env["REAL_GIT"],
            "-C",
            str(repo),
            "remote",
            "set-url",
            "origin",
            "https://offline.invalid/repo",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / ".github/scripts/ci_result_source.py"),
            "prepare",
            "--repo",
            str(repo),
            "--workflow-sha",
            workflow,
            "--directory",
            str(tmp_path / "task"),
            "--github-env",
            env["GITHUB_ENV"],
        ],
        env=env,
        check=True,
    )
    # Recording must work without any repository retry or selection helper.
    (repo / "ci/npu_ops_test/changes.py").unlink()
    assert not calls.exists()
    values = dict(
        line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines()
    )
    env.update(
        values,
        WORK_DIR=str(repo),
        EXPECTED_SOURCE=candidate,
        CONFIG="test/ci/ut/ut-runtime-1npu.yaml",
        RUNNER="tokenspeed-npu-arm64",
    )
    record = npu_step_script(
        "run-ci-task-matrix.yml", "test", "Record NPU checked-out source"
    )
    result = subprocess.run(
        ["bash", "-c", record], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads((repo / ".ci-artifacts/source.json").read_text())["source_sha"]
        == candidate
    )
    assert not calls.exists()
    (repo / ".ci-artifacts/source.json").unlink()
    env["EXPECTED_SOURCE"] = "0" * 40
    mismatch = subprocess.run(
        ["bash", "-c", record], env=env, text=True, capture_output=True
    )
    assert mismatch.returncode != 0
    assert not (repo / ".ci-artifacts/source.json").exists()
    reporter = Path(values["NPU_SOURCE_REPORTER"])
    reporter.chmod(0o600)
    reporter.write_text("raise RuntimeError('changed')\n")
    changed = subprocess.run(
        ["bash", "-c", record], env=env, text=True, capture_output=True
    )
    assert changed.returncode != 0
    assert "Prepared source reporter was modified" in changed.stderr


@pytest.mark.parametrize("status, proof_status", [(0, 0), (7, 0), (0, 7)])
def test_runtime_install_entry_preserves_interpreter_arguments_and_status(
    tmp_path, status, proof_status
):
    from pipeline import normalize_task

    task = normalize_task(REPO_ROOT / "test/ci/ut/ut-runtime-1npu.yaml", REPO_ROOT)
    interpreter = tmp_path / "bootstrap python"
    arguments = tmp_path / "arguments.json"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + f"import json, pathlib, sys\npathlib.Path({str(arguments)!r}).write_text(json.dumps(sys.argv[1:]))\nsys.exit({status})\n"
    )
    interpreter.chmod(0o755)
    proof_python = tmp_path / "model-venv/bin/python"
    proof_python.parent.mkdir(parents=True)
    proof_python.write_text(
        f"#!{sys.executable}\nimport sys\nsys.exit({proof_status})\n"
    )
    proof_python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", task["install"][0]],
        env=dict(
            os.environ,
            NPU_CI_BOOTSTRAP_PYTHON=str(interpreter),
            NPU_CI_WORK=str(tmp_path),
            CI_RUNNER_LABEL="tokenspeed-npu-arm64",
            NPU_CI_RUNTIME="true",
        ),
        cwd=REPO_ROOT,
    )
    assert result.returncode == (status or proof_status)
    assert json.loads(arguments.read_text()) == ["-B", "-"]


def npu_installer_functions():
    source = (REPO_ROOT / "test/ci_system/install_deps.sh").read_text()
    script = source.split("<<'NPU_INSTALL'\n", 1)[1].split("\nNPU_INSTALL", 1)[0]
    namespace = {"__name__": "npu_installer_test"}
    exec(compile(script, "install_deps.sh:NPU_INSTALL", "exec"), namespace)
    return namespace


def write_npu_wheel_manifest(directory):
    import hashlib

    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.glob("*.whl")
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": files}))
    manifest.chmod(0o644)


@pytest.mark.parametrize(
    "damage",
    ["content", "extra", "symlink", "writable-file", "writable-directory", "owner"],
)
def test_npu_wheelhouse_rejects_untrusted_changes(tmp_path, damage):
    wheel = tmp_path / "fixture-1-py3-none-any.whl"
    wheel.write_bytes(b"verified fixture")
    wheel.chmod(0o644)
    tmp_path.chmod(0o755)
    write_npu_wheel_manifest(tmp_path)
    verify = npu_installer_functions()["verify_wheelhouse"]
    result = verify(tmp_path, owner_uid=os.getuid())
    assert result["wheel_count"] == 1
    if damage == "content":
        wheel.write_bytes(b"changed")
    elif damage == "extra":
        (tmp_path / "extra-1-py3-none-any.whl").write_bytes(b"extra")
    elif damage == "symlink":
        data = wheel.read_bytes()
        wheel.unlink()
        target = tmp_path.parent / "outside.whl"
        target.write_bytes(data)
        wheel.symlink_to(target)
    elif damage == "writable-file":
        wheel.chmod(0o666)
    elif damage == "writable-directory":
        tmp_path.chmod(0o777)
    owner = os.getuid() + 1 if damage == "owner" else os.getuid()
    with pytest.raises(ValueError):
        verify(tmp_path, owner_uid=owner)


@pytest.mark.parametrize("missing_dependency", [False, True])
def test_npu_dependency_install_uses_wheels_without_index(
    tmp_path, monkeypatch, missing_dependency
):
    import threading
    import venv
    import zipfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class Index(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_error(503)

        def log_message(self, *args):
            pass

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir(mode=0o755)
    name = "npu_ci_offline_fixture"
    info = f"{name}-1.0.dist-info"
    files = {
        f"{name}.py": "VALUE = 42\n",
        f"{info}/METADATA": (
            "Metadata-Version: 2.1\nName: npu-ci-offline-fixture\nVersion: 1.0\n"
            + (
                "Requires-Dist: npu-ci-missing-dependency>=1\n"
                if missing_dependency
                else ""
            )
        ),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{info}/RECORD"] = (
        "".join(f"{path},,\n" for path in files) + f"{info}/RECORD,,\n"
    )
    wheel = wheelhouse / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    wheel.chmod(0o644)
    write_npu_wheel_manifest(wheelhouse)
    installer = npu_installer_functions()
    installer["verify_wheelhouse"](wheelhouse, owner_uid=os.getuid())
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = str(environment / "bin/python")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Index)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_INDEX_URL", f"http://127.0.0.1:{server.server_port}/simple")
    monkeypatch.setenv(
        "PIP_EXTRA_INDEX_URL", f"http://127.0.0.1:{server.server_port}/extra"
    )
    for key in ("PIP_NO_INDEX", "PIP_FIND_LINKS", "PIP_CONSTRAINT"):
        monkeypatch.delenv(key, raising=False)
    try:

        def install():
            installer["install_from_wheelhouse"](
                python,
                wheelhouse,
                ["npu-ci-offline-fixture==1.0"],
                options=["--only-binary=:all:", "--disable-pip-version-check"],
            )

        if missing_dependency:
            with pytest.raises(subprocess.CalledProcessError):
                install()
        else:
            install()
            output = subprocess.check_output(
                [python, "-c", f"import {name}; print({name}.VALUE)"], text=True
            )
            assert output.strip() == "42"
            subprocess.run([python, "-m", "pip", "check"], check=True)
        assert requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "allowed_names,runner_name,accepted",
    [
        ("npu-one", "npu-one", True),
        ("npu-one\nnpu-two", "npu-two", True),
        ("\nnpu-one\r\n\nnpu-two\r\n", "npu-one", True),
        ("", "npu-one", False),
        ("npu-one\nnpu-two", "npu-other", False),
        ("npu-one-more", "npu-one", False),
        ("npu-*", "npu-one", False),
        ("npu-one", "", False),
    ],
)
def test_npu_runner_allowlist_checks_exact_names(allowed_names, runner_name, accepted):
    script = npu_step_script(
        "run-ci-task-matrix.yml", "test", "Check NPU runner environment"
    )
    # Stop at the next real guard, before Linux/ARM checks or filesystem changes.
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "NPU_RUNNER_NAMES": allowed_names,
            "RUNNER_NAME": runner_name,
            "GITHUB_SERVER_URL": "https://unsupported.invalid",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    if accepted:
        assert "Unsupported GitHub server" in result.stderr
        assert "NPU runner name does not match" not in result.stderr
    else:
        assert "NPU runner name does not match the deployment" in result.stderr
        assert "Unsupported GitHub server" not in result.stderr


def test_npu_pool_schedules_each_suite_once():
    matrix = build_matrix(
        REPO_ROOT / "test/ci", REPO_ROOT, "per-commit", "npu", "unit-test"
    )
    assert Counter(entry["name"] for entry in matrix["include"]) == {
        "npu-operators": 1,
        "ut-runtime-1npu": 1,
    }
    assert {entry["runner"] for entry in matrix["include"]} == {"tokenspeed-npu-arm64"}
