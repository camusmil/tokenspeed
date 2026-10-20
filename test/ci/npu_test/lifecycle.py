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
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
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
    for key in ("jobs_root", "evidence_root", "state_root", "diag_root"):
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
        "success_seconds",
        "failure_seconds",
        "diagnostic_seconds",
        "minimum_free_bytes",
        "pending_warning_seconds",
    ):
        require(type(policy[key]) is int and policy[key] >= 0, f"Invalid policy: {key}")
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


def collect(policy, work, now):
    work, state = checked_work(policy, work)
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
        for name in ("bootstrap.log", "fetch.tsv", "source.txt"):
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
                relative = path.relative_to(directory)
                if len(relative.parts) == 1 and path.name in PHASE_FILES:
                    selected.append(path)
                elif len(relative.parts) == 2 and re.fullmatch(
                    r"[A-Za-z0-9_-]+", relative.parts[0]
                ):
                    if path.name in OPERATOR_FILES | {f"{phase}.log", f"{phase}.rc"}:
                        selected.append(path)
        entries = []
        for source in sorted(selected):
            require(stat.S_ISREG(source.lstat().st_mode), "Invalid evidence file")
            target = temporary / source.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # The evidence and jobs roots share a filesystem on the host.
                # Hard links avoid storing a second copy of raw operator logs.
                os.link(source, target, follow_symlinks=False)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                with os.fdopen(
                    os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb"
                ) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
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
    return receipt


def open_diagnostic_files():
    opened = set()
    try:
        for process in Path("/proc").iterdir():
            if not process.name.isdigit() or process.stat().st_uid != os.getuid():
                continue
            try:
                for fd in (process / "fd").iterdir():
                    try:
                        info = fd.stat()
                        opened.add((info.st_dev, info.st_ino))
                    except FileNotFoundError:
                        pass
            except FileNotFoundError:
                pass
    except (OSError, PermissionError):
        return None
    return opened


def garbage_collect(policy, mode, now):
    require(mode in ("dry-run", "apply"), "Explicit cleanup mode is required")
    result = {
        "at": utc(now),
        "mode": mode,
        "removed": [],
        "eligible": [],
        "protected": [],
        "unmanaged": 0,
        "errors": [],
    }
    for work in Path(policy["jobs_root"]).iterdir():
        if not (work / "lifecycle.json").is_file() or work.is_symlink():
            result["unmanaged"] += 1
            continue
        try:
            _, state = checked_work(policy, work)
            result["protected"].append(
                {
                    "run_id": state["run_id"],
                    "reason": (
                        "active-or-unknown-owner"
                        if owner_status(state["owner"]) != "gone"
                        else "archive-not-confirmed"
                    ),
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            result["errors"].append(str(exc))
    for output in Path(policy["evidence_root"]).iterdir():
        try:
            require(
                output == output.resolve()
                and output.is_dir()
                and re.fullmatch(r"[0-9]+-[0-9]+", output.name),
                "Invalid evidence directory",
            )
            state = read_json(output / "lifecycle.json")
            require(
                state["schema"] == SCHEMA
                and state["repository"] == policy["repository"],
                "Evidence repository mismatch",
            )
            require(
                output == evidence_directory(policy, state),
                "Evidence run identity mismatch",
            )
            if owner_status(state["owner"]) != "gone":
                result["protected"].append(
                    {"run_id": state["run_id"], "reason": "active-or-unknown-owner"}
                )
                continue
            receipt_path = output / "receipt.json"
            if (
                not receipt_path.exists()
                or read_json(receipt_path).get("uploaded") is not True
            ):
                result["protected"].append(
                    {"run_id": state["run_id"], "reason": "archive-not-confirmed"}
                )
                continue
            receipt = read_json(receipt_path)
            require(
                re.fullmatch(r"[0-9]+", receipt["artifact_id"])
                and re.fullmatch(r"[0-9a-f]{64}", receipt["artifact_digest"]),
                "Invalid upload receipt",
            )
            retention = (
                policy["success_seconds"]
                if receipt["accuracy"] == receipt["performance"] == "success"
                else policy["failure_seconds"]
            )
            if now - receipt["at"] < retention:
                continue
            verify_bundle(output / "bundle")
            work = Path(state["work"])
            match = RUN_NAME.fullmatch(work.name)
            require(
                work.parent == Path(policy["jobs_root"]) and match is not None,
                "Evidence points outside the slot",
            )
            require(
                list(match.groups()) == [state["run_id"], state["attempt"]],
                "Evidence points at another run",
            )
            if work.exists():
                _, current = checked_work(policy, work)
                require(current["owner"] == state["owner"], "Work owner changed")
                require(
                    owner_status(current["owner"]) == "gone", "Work owner is active"
                )
            result["eligible"].append(output.name)
            if mode == "apply":
                if work.exists():
                    remove_directory(work, policy["jobs_root"])
                remove_directory(output, policy["evidence_root"])
                result["removed"].append(output.name)
        except (OSError, ValueError, KeyError) as exc:
            result["errors"].append(f"{output.name}: {exc}")
    pending = set()
    for work in Path(policy["jobs_root"]).iterdir():
        if work.is_symlink() or not (work / "lifecycle.json").is_file():
            continue
        try:
            _, state = checked_work(policy, work)
            if (
                owner_status(state["owner"]) == "gone"
                and now - state["started_at"] >= policy["pending_warning_seconds"]
            ):
                pending.add(state["run_id"])
        except (OSError, ValueError, KeyError):
            continue
    result["overdue_unarchived_runs"] = sorted(pending)
    if pending:
        result["errors"].append(
            f"Unconfirmed archives older than retention warning threshold: {len(pending)}"
        )
    opened = open_diagnostic_files()
    if opened is not None:
        for path in Path(policy["diag_root"]).glob("*.log"):
            try:
                info = path.lstat()
                if not re.fullmatch(
                    r"(?:Worker|Runner)_[0-9]{8}-[0-9]{6}-utc\.log", path.name
                ) or not stat.S_ISREG(info.st_mode):
                    continue
                if (
                    now - info.st_mtime < policy["diagnostic_seconds"]
                    or (info.st_dev, info.st_ino) in opened
                ):
                    continue
                result["eligible"].append(path.name)
                if mode == "apply":
                    path.unlink()
                    result["removed"].append(path.name)
            except OSError as exc:
                result["errors"].append(str(exc))
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
        workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
        action = workspace / ".npu-ci-archive"
        if action.is_symlink():
            action.unlink()
        elif action.exists():
            remove_directory(action, workspace)
        shutil.copytree(Path(__file__).parent / "archive", action)
        work = initialize(policy, identity, worker_identity(), now)
        with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
            handle.write(f"NPU_CI_WORK={work}\n")
        print(work)
        previous = Path(policy["state_root"]) / "last-gc.json"
        if previous.exists():
            status = read_json(previous)
            append_summary(
                f"\n### Host cleanup\n\nLast run: `{status['at']}`; removed: `{len(status['removed'])}`; errors: `{len(status['errors'])}`.\n"
            )
            if status["errors"]:
                print(
                    "::warning::Previous host cleanup reported errors; inspect npu-ci-retention.service",
                    file=sys.stderr,
                )
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
        stream_bootstrap(args.work, sys.stdin.buffer, sys.stdout.buffer)
    elif args.command == "collect":
        bundle = collect(policy, args.work, now)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"path={bundle}\n")
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
        return 1 if result["errors"] else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(cli())
    except Exception as exc:
        message = f"NPU evidence/retention failed: {type(exc).__name__}: {exc}"
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}", file=sys.stderr)
        raise SystemExit(1)
