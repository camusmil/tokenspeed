# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Manage NPU evidence and retention outside the checked-out test sources."""

import argparse
import csv
import hashlib
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 2
RUN_NAME = re.compile(r"actions-([0-9]+)-([0-9]+)\.[A-Za-z0-9_]+")
PHASE_FILES = {"context.json", "report.json", "summary.md", "executor-error.log"}
OPERATOR_FILES = {"process.json", "result.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), f"Not a regular file: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "r") as handle:
        text = handle.read(1024 * 1024 + 1)
    require(len(text) <= 1024 * 1024, f"Metadata too large: {path}")
    return json.loads(text)


def write_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), f"Refusing metadata symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc(timestamp):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def load_policy(path):
    policy = read_json(path)
    require(
        policy.get("schema_version") == 1 and policy.get("delete_unarchived") is False,
        "Unarchived deletion is not permitted",
    )
    for key in ("jobs_root", "evidence_root", "state_root"):
        directory = Path(policy[key])
        require(
            directory.is_absolute() and directory == directory.resolve(),
            f"Unsafe policy path: {key}",
        )
        require(directory.is_dir(), f"Deployment directory missing: {directory}")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", policy["repository"]),
        "Invalid repository",
    )
    for key in (
        "artifact_retention_days",
        "artifact_max_bytes",
        "upload_attempts",
        "minimum_free_bytes",
        "pending_warning_seconds",
    ):
        require(type(policy[key]) is int and policy[key] >= 0, f"Invalid policy: {key}")
    require(1 <= policy["artifact_retention_days"] <= 90, "Invalid artifact retention")
    require(1 <= policy["upload_attempts"] <= 10, "Invalid upload attempts")
    require(policy["artifact_max_bytes"] > 0, "Invalid artifact size limit")
    return policy


def process_identity(pid):
    directory = Path(f"/proc/{pid}")
    # Missing /proc on a non-Linux test host is treated conservatively.
    if not Path("/proc").is_dir():
        return {"pid": pid, "boot_id": "unknown", "start_ticks": "unknown"}
    text = (directory / "stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": fields[19],
    }


def worker_identity():
    pid = os.getpid()
    for _ in range(30):
        directory = Path(f"/proc/{pid}")
        if (directory / "comm").read_text().strip() == "Runner.Worker":
            return process_identity(pid)
        text = (directory / "stat").read_text()
        pid = int(text[text.rfind(")") + 2 :].split()[1])
        if pid <= 1:
            break
    raise RuntimeError(
        "Runner.Worker ancestor not found; refusing an unowned CI directory"
    )


def owner_status(owner):
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return "gone"
    except (PermissionError, ValueError, KeyError):
        return "unknown"
    try:
        actual = process_identity(int(owner["pid"]))
        if Path("/proc").is_dir():
            text = Path(f"/proc/{owner['pid']}/stat").read_text()
            if text[text.rfind(")") + 2 :].split()[0] == "Z":
                return "gone"
    except FileNotFoundError:
        return "gone"
    except (OSError, ValueError):
        return "unknown"
    if "unknown" in (actual["boot_id"], owner.get("boot_id")):
        return "unknown"
    return "alive" if actual == owner else "gone"


def checked_work(policy, path):
    work = Path(path)
    require(
        work.parent == Path(policy["jobs_root"]),
        "Work directory is outside the configured slot",
    )
    match = RUN_NAME.fullmatch(work.name)
    require(
        match is not None and work == work.resolve() and work.is_dir(),
        "Invalid work directory",
    )
    state = read_json(work / "lifecycle.json")
    require(
        state["schema"] == SCHEMA and state["repository"] == policy["repository"],
        "Run policy mismatch",
    )
    require(
        [state["run_id"], state["attempt"]] == list(match.groups()),
        "Run identity mismatch",
    )
    require(state["work"] == str(work), "Work identity mismatch")
    return work, state


def initialize(policy, identity, owner, now):
    require(
        identity["repository"] == policy["repository"],
        "Repository does not match host policy",
    )
    for name in ("run_id", "attempt"):
        require(re.fullmatch(r"[0-9]+", identity[name]), f"Invalid {name}")
    require(re.fullmatch(r"[0-9a-f]{40}", identity["sha"]), "Invalid event SHA")
    require(
        shutil.disk_usage(policy["jobs_root"]).free >= policy["minimum_free_bytes"],
        "Insufficient free disk space for NPU CI",
    )
    work = Path(
        tempfile.mkdtemp(
            prefix=f"actions-{identity['run_id']}-{identity['attempt']}.",
            dir=policy["jobs_root"],
        )
    )
    state = dict(
        identity, schema=SCHEMA, work=str(work), owner=owner, started_at=now, phases={}
    )
    write_json(work / "lifecycle.json", state)
    return work


def phase_state(policy, work, phase, stage, rc, now):
    require(phase in ("accuracy", "performance"), "Unknown phase")
    work, state = checked_work(policy, work)
    state["phases"][phase] = {"stage": stage, "exit": rc, "at": now}
    write_json(work / "lifecycle.json", state)


def regular_files(directory):
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            require(
                not (Path(parent) / name).is_symlink(),
                "Evidence must not contain symlinks",
            )
        for name in files:
            path = Path(parent) / name
            require(
                stat.S_ISREG(path.lstat().st_mode),
                "Evidence must contain regular files only",
            )
            yield path


def digest(path):
    value = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def evidence_directory(policy, state):
    return Path(policy["evidence_root"]) / f"{state['run_id']}-{state['attempt']}"


def collect(policy, work, now, outcomes):
    work, state = checked_work(policy, work)
    for directory in (
        work / "accuracy",
        work / "performance",
        work / "plan-collection",
    ):
        if directory.exists():
            for record in directory.rglob("*.owner.json"):
                for identity in read_json(record).values():
                    require(
                        owner_status(identity) == "gone",
                        "Evidence producer is still active; refusing collection",
                    )
    logger = work / "bootstrap.owner.json"
    if logger.exists():
        require(
            owner_status(read_json(logger)) == "gone",
            "Bootstrap logger is still active",
        )
    output = evidence_directory(policy, state)
    require(not output.is_symlink(), "Invalid evidence directory")
    output.mkdir(exist_ok=True)
    bundle = output / "bundle"
    if bundle.exists():
        verify_bundle(bundle)
        return bundle
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=output))
    try:
        selected = []
        require(set(outcomes) == {"accuracy", "performance"}, "Missing step outcomes")
        succeeded = all(value == "success" for value in outcomes.values()) and all(
            state["phases"].get(phase, {}).get("exit") == 0
            for phase in ("accuracy", "performance")
        )
        for name in (
            ()
            if succeeded
            else ("bootstrap.log", "fetch.tsv", "source.txt", "frozen-plan.json")
        ):
            if (work / name).exists():
                require(
                    not (work / name).is_symlink(), "Evidence must not contain symlinks"
                )
                selected.append(work / name)
        for phase in ("accuracy", "performance"):
            directory = work / phase
            if state["phases"].get(phase, {}).get("exit") == 0:
                for name in ("context.json", "report.json", "summary.md"):
                    require(
                        (directory / name).is_file(),
                        f"Required phase evidence missing: {phase}/{name}",
                    )
            if not directory.exists():
                continue
            require(directory == directory.resolve(), "Invalid phase directory")
            for path in regular_files(directory):
                if succeeded and path.name not in ("context.json", "summary.md"):
                    continue
                relative = path.relative_to(directory)
                if len(relative.parts) == 1 and path.name in PHASE_FILES:
                    selected.append(path)
                elif len(relative.parts) == 2 and re.fullmatch(
                    r"[A-Za-z0-9_-]+", relative.parts[0]
                ):
                    if path.name in OPERATOR_FILES | {f"{phase}.log", f"{phase}.rc"}:
                        selected.append(path)
        require(selected, "No evidence available for upload")
        total = sum(source.stat().st_size for source in selected)
        require(
            total <= policy["artifact_max_bytes"],
            "Evidence exceeds artifact size limit; retained locally",
        )
        entries = []
        for source in sorted(selected):
            require(stat.S_ISREG(source.lstat().st_mode), "Invalid evidence file")
            target = temporary / source.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            # These roots must share a filesystem; one inode holds each raw log.
            os.link(source, target, follow_symlinks=False)
            entries.append(
                {
                    "path": str(target.relative_to(temporary)),
                    "bytes": target.stat().st_size,
                    "sha256": digest(target),
                }
            )
        write_json(
            temporary / "run.json",
            {
                key: state[key]
                for key in (
                    "schema",
                    "repository",
                    "run_id",
                    "attempt",
                    "sha",
                    "ref",
                    "runner",
                    "started_at",
                    "phases",
                )
            },
        )
        entries.append(
            {
                "path": "run.json",
                "bytes": (temporary / "run.json").stat().st_size,
                "sha256": digest(temporary / "run.json"),
            }
        )
        write_json(
            temporary / "manifest.json",
            {"schema": SCHEMA, "collected_at": utc(now), "files": entries},
        )
        require(
            sum(path.stat().st_size for path in regular_files(temporary))
            <= policy["artifact_max_bytes"],
            "Evidence metadata exceeds artifact size limit",
        )
        os.rename(temporary, bundle)
        write_json(output / "lifecycle.json", state)
        return bundle
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_bundle(bundle):
    require(bundle == bundle.resolve(), "Invalid evidence bundle path")
    manifest = read_json(bundle / "manifest.json")
    require(
        manifest["schema"] == SCHEMA and manifest["files"], "Invalid evidence manifest"
    )
    expected_files = {item["path"] for item in manifest["files"]}
    require(len(expected_files) == len(manifest["files"]), "Duplicate evidence file")
    require(
        {str(path.relative_to(bundle)) for path in regular_files(bundle)}
        == expected_files | {"manifest.json"},
        "Unexpected evidence files",
    )
    for item in manifest["files"]:
        relative = Path(item["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid manifest path",
        )
        path = bundle / relative
        require(
            path == path.resolve() and stat.S_ISREG(path.lstat().st_mode),
            "Invalid manifest file",
        )
        require(
            path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
            f"Evidence integrity failure: {relative}",
        )


def remove_directory(path, parent):
    path, parent = Path(path), Path(parent)
    require(
        path.parent == parent and path == path.resolve() and path.is_dir(),
        "Refusing unsafe deletion",
    )
    require(
        shutil.rmtree.avoids_symlink_attacks, "Platform lacks safe recursive deletion"
    )
    shutil.rmtree(path)


def acknowledge(
    policy, work, outcome, artifact_id, artifact_digest, accuracy, performance, now
):
    work, state = checked_work(policy, work)
    output = evidence_directory(policy, state)
    require(
        output == output.resolve() and output.is_dir(),
        "Evidence collection did not finish",
    )
    verify_bundle(output / "bundle")
    uploaded = (
        outcome == "success"
        and bool(re.fullmatch(r"[0-9]+", artifact_id))
        and bool(re.fullmatch(r"[0-9a-f]{64}", artifact_digest))
    )
    receipt = {
        "schema": SCHEMA,
        "uploaded": uploaded,
        "at": now,
        "accuracy": accuracy,
        "performance": performance,
        "artifact_id": artifact_id,
        "artifact_digest": artifact_digest,
        "work_removed": False,
    }
    write_json(output / "receipt.json", receipt)
    if not uploaded:
        return receipt
    # Only this job's finalization explicitly deletes its own active work area.
    # The timer always waits for the owning Runner.Worker to disappear.
    remove_directory(work, policy["jobs_root"])
    receipt["work_removed"] = True
    write_json(output / "receipt.json", receipt)
    # Preserve the minimal receipt outside the disposable evidence directory.
    write_json(
        Path(policy["state_root"]) / f"{state['run_id']}-{state['attempt']}.json",
        dict(
            receipt,
            repository=state["repository"],
            sha=state["sha"],
            run_id=state["run_id"],
            attempt=state["attempt"],
        ),
    )
    remove_directory(output, policy["evidence_root"])
    receipt["local_evidence_removed"] = True
    return receipt


def garbage_collect(policy, mode, now):
    """Report abandoned unarchived runs; never destroy their only evidence."""
    result = {
        "at": utc(now),
        "mode": mode,
        "pending": [],
        "unmanaged": [],
        "errors": [],
    }
    for work in Path(policy["jobs_root"]).iterdir():
        if not RUN_NAME.fullmatch(work.name):
            result["unmanaged"].append(work.name)
            continue
        try:
            _, state = checked_work(policy, work)
            status = owner_status(state["owner"])
            if (
                status == "gone"
                and now - state["started_at"] >= policy["pending_warning_seconds"]
            ):
                result["pending"].append(
                    {
                        "work": work.name,
                        "reason": "unarchived-evidence-requires-operator-action",
                    }
                )
        except Exception as exc:
            result["errors"].append({"work": work.name, "error": str(exc)})
    if mode == "apply":
        write_json(Path(policy["state_root"]) / "last-gc.json", result)
    return result


def stream_bootstrap(work, source, console):
    path = Path(work) / "bootstrap.log"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as log:
        for line in source:
            stamp = (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            log.write(stamp.encode() + b" " + line)
            log.flush()
            console.write(line)
            console.flush()
        os.fsync(log.fileno())


def append_summary(text):
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(text)


def supervise(command, parent_fd):
    """Keep a test group owned after controller/Runner death, including SIGKILL."""
    require(sys.platform == "linux", "Process supervision requires Linux")
    interrupted = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: interrupted.append(number))
    owner = worker_identity() if os.environ.get("GITHUB_ACTIONS") == "true" else None
    lock_fds = ()
    try:
        os.fstat(9)
        lock_fds = (9,)
    except OSError:
        pass
    process = subprocess.Popen(command, start_new_session=True, pass_fds=lock_fds)
    reason = None
    try:
        record = os.environ.get("NPU_PROCESS_RECORD")
        if record:
            write_json(
                record,
                {
                    "supervisor": process_identity(os.getpid()),
                    "test": process_identity(process.pid),
                },
            )
        while True:
            # WNOWAIT keeps the group leader's PID reserved until group cleanup.
            if (
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is not None
            ):
                break
            if interrupted:
                reason = f"supervisor interrupted: signal={interrupted[0]}"
                break
            ready, _, _ = select.select([parent_fd], [], [], 0.1)
            if ready and not os.read(parent_fd, 1):
                reason = "phase controller exited; reclaiming its test group"
                break
            if owner is not None and owner_status(owner) != "alive":
                reason = "Runner worker exited; reclaiming its test group"
                break
    finally:
        if reason:
            try:
                print(reason, file=sys.stderr, flush=True)
            except BrokenPipeError:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(0.2)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        os.close(parent_fd)
    if reason:
        return 128 + (interrupted[0] if interrupted else signal.SIGTERM)
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def prepare_archive_action(workspace, policy):
    """Generate the one finalization step from the trusted host runtime."""
    action = Path(workspace).resolve() / ".npu-ci-archive-phase1"
    if action.is_symlink():
        action.unlink()
    elif action.exists():
        remove_directory(action, action.parent)
    action.mkdir()
    helper = '/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B "$NPU_CI_RUNTIME/ci_lifecycle.py" --policy "$NPU_CI_RUNTIME/policy.json"'
    steps = [
        {
            "name": "Collect NPU evidence",
            "id": "collect",
            "shell": "bash",
            "env": {
                "NPU_ARCHIVE_WORK": "${{ inputs.work }}",
                "NPU_ACCURACY_OUTCOME": "${{ inputs.accuracy }}",
                "NPU_PERFORMANCE_OUTCOME": "${{ inputs.performance }}",
            },
            "run": helper
            + ' collect --work "$NPU_ARCHIVE_WORK" --accuracy "$NPU_ACCURACY_OUTCOME" --performance "$NPU_PERFORMANCE_OUTCOME"',
        }
    ]
    for attempt in range(1, policy["upload_attempts"] + 1):
        earlier_failed = "".join(
            f" && steps.upload{i}.outcome != 'success'" for i in range(1, attempt)
        )
        steps.append(
            {
                "name": f"Upload NPU evidence (attempt {attempt}/{policy['upload_attempts']})",
                "id": f"upload{attempt}",
                "if": "${{ always() && steps.collect.outcome == 'success'"
                + earlier_failed
                + " }}",
                "continue-on-error": True,
                "uses": "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9",
                "with": {
                    "name": "npu-evidence-${{ github.run_id }}-${{ github.run_attempt }}-upload-"
                    + str(attempt),
                    "path": "${{ steps.collect.outputs.path }}",
                    "retention-days": policy["artifact_retention_days"],
                    "if-no-files-found": "error",
                    "include-hidden-files": False,
                },
            }
        )
    successful = " || ".join(
        f"steps.upload{i}.outcome == 'success'"
        for i in range(1, policy["upload_attempts"] + 1)
    )
    fields = {}
    for field in ("artifact-id", "artifact-digest"):
        value = " || ".join(
            f"(steps.upload{i}.outcome == 'success' && steps.upload{i}.outputs.{field})"
            for i in range(1, policy["upload_attempts"] + 1)
        )
        fields[field] = "${{ " + value + " || '' }}"
    steps.append(
        {
            "name": "Confirm archive and remove this run's files",
            "if": "${{ always() && steps.collect.outcome == 'success' }}",
            "shell": "bash",
            "env": {
                "NPU_ARCHIVE_WORK": "${{ inputs.work }}",
                "NPU_ACCURACY_OUTCOME": "${{ inputs.accuracy }}",
                "NPU_PERFORMANCE_OUTCOME": "${{ inputs.performance }}",
                "NPU_UPLOAD_OUTCOME": "${{ ("
                + successful
                + ") && 'success' || 'failure' }}",
                "NPU_ARTIFACT_ID": fields["artifact-id"],
                "NPU_ARTIFACT_DIGEST": fields["artifact-digest"],
            },
            "run": helper
            + ' ack --work "$NPU_ARCHIVE_WORK" --outcome "$NPU_UPLOAD_OUTCOME" --artifact-id "$NPU_ARTIFACT_ID" --artifact-digest "$NPU_ARTIFACT_DIGEST" --accuracy "$NPU_ACCURACY_OUTCOME" --performance "$NPU_PERFORMANCE_OUTCOME"',
        }
    )
    value = {
        "name": "Archive NPU CI evidence",
        "description": "Upload selected evidence before removing this run's disposable files.",
        "inputs": {
            name: {"description": name, "required": True}
            for name in ("work", "accuracy", "performance")
        },
        "runs": {"using": "composite", "steps": steps},
    }
    (action / "action.yml").write_text(json.dumps(value, indent=2) + "\n")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    phase = commands.add_parser("phase")
    phase.add_argument("--work", required=True)
    phase.add_argument("--phase", choices=("accuracy", "performance"), required=True)
    phase.add_argument("--stage", required=True)
    phase.add_argument("--exit", type=int, required=True)
    logger = commands.add_parser("log")
    logger.add_argument("--work", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--work", required=True)
    collector.add_argument("--accuracy", required=True)
    collector.add_argument("--performance", required=True)
    ack = commands.add_parser("ack")
    for name in (
        "work",
        "outcome",
        "artifact-id",
        "artifact-digest",
        "accuracy",
        "performance",
    ):
        ack.add_argument(f"--{name}", required=True)
    gc = commands.add_parser("gc")
    gc.add_argument("--mode", choices=("dry-run", "apply"), required=True)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    now = time.time()
    if args.command in ("phase", "collect", "ack"):
        _, owned = checked_work(policy, args.work)
        require(owned["owner"] == worker_identity(), "Another worker owns this run")
        require(
            owned["run_id"] == os.environ["GITHUB_RUN_ID"]
            and owned["attempt"] == os.environ["GITHUB_RUN_ATTEMPT"],
            "Another job owns this run",
        )
    if args.command == "init":
        identity = {
            name: os.environ[variable]
            for name, variable in (
                ("repository", "GITHUB_REPOSITORY"),
                ("run_id", "GITHUB_RUN_ID"),
                ("attempt", "GITHUB_RUN_ATTEMPT"),
                ("sha", "GITHUB_SHA"),
                ("ref", "GITHUB_REF"),
                ("runner", "RUNNER_NAME"),
            )
        }
        prepare_archive_action(os.environ["GITHUB_WORKSPACE"], policy)
        work = initialize(policy, identity, worker_identity(), now)
        with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
            handle.write(f"NPU_CI_WORK={work}\n")
        print(work)
    elif args.command == "phase":
        phase_state(policy, args.work, args.phase, args.stage, args.exit, now)
        if args.phase == "accuracy" and (Path(args.work) / "fetch.tsv").exists():
            with (Path(args.work) / "fetch.tsv").open() as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            append_summary(
                "\n### Source download\n\n| Attempt | Started (UTC) | Finished (UTC) | Seconds | Exit |\n| --- | --- | --- | --- | --- |\n"
                + "".join("| " + " | ".join(row) + " |\n" for row in rows)
            )
    elif args.command == "log":
        checked_work(policy, args.work)
        write_json(
            Path(args.work) / "bootstrap.owner.json", process_identity(os.getpid())
        )
        stream_bootstrap(args.work, sys.stdin.buffer, sys.stdout.buffer)
    elif args.command == "collect":
        bundle = collect(
            policy,
            args.work,
            now,
            {"accuracy": args.accuracy, "performance": args.performance},
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"path={bundle}\n")
        print(
            json.dumps(
                {
                    "path": str(bundle),
                    "files": [str(p) for p in regular_files(bundle)],
                    "retention_days": policy["artifact_retention_days"],
                    "upload_attempts": policy["upload_attempts"],
                }
            )
        )
    elif args.command == "ack":
        receipt = acknowledge(
            policy,
            args.work,
            args.outcome,
            args.artifact_id,
            args.artifact_digest,
            args.accuracy,
            args.performance,
            now,
        )
        link = f"https://github.com/{policy['repository']}/actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts/{args.artifact_id}"
        append_summary(
            f"\n### Evidence archive\n\nAccuracy: `{args.accuracy}`; performance: `{args.performance}`.\n\nUpload confirmed: `{receipt['uploaded']}`; work directory removed: `{receipt['work_removed']}`.\n"
            + (
                f"\n[Download evidence]({link})\n\nSHA-256: `{args.artifact_digest}`\n"
                if receipt["uploaded"]
                else "\nUpload failed. All local evidence and work files have been retained.\n"
            )
        )
        return 0 if receipt["uploaded"] and receipt["work_removed"] else 1
    else:
        import fcntl

        with (Path(policy["state_root"]) / "gc.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another cleanup is active; skipped")
                return 0
            result = garbage_collect(policy, args.mode, now)
        print(json.dumps(result, indent=2))
        if result["pending"]:
            print(
                "::warning::Unarchived NPU evidence is older than 24 hours; manual handling required"
            )
        return 1 if result["errors"] or result["pending"] else 0
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "supervise":
        raise SystemExit(supervise(sys.argv[3:], int(sys.argv[2])))
    try:
        raise SystemExit(cli())
    except Exception as exc:
        message = f"NPU evidence/retention failed: {type(exc).__name__}: {exc}"
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}", file=sys.stderr)
        raise SystemExit(1)
