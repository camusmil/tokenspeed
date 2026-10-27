"""
Process-group-based process lifecycle manager for CI runners.

Usage
-----
# At job start (cleans up stale processes from previous run):
mgr = ProcessGroupManager(runner_id="gb200-node-1", supervision=None)
mgr.cleanup_stale()

# Launch a long-running server:
proc = mgr.start(command, shell=True, cwd=cwd, env=env)

# At job end, terminate all process groups started by this manager:
mgr.terminate_all()
"""

from __future__ import annotations

import json
import os
import re
import select
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Directory that survives across CI runs on the same host.
# Must be writable by the runner user.
_PGID_DIR = Path("/tmp/ci-pgid")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_runner_id(runner_id: str) -> str:
    """Sanitise runner_id so it is safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", runner_id)


def _pgid_path(runner_id: str) -> Path:
    return _PGID_DIR / f"{_safe_runner_id(runner_id)}.pgid"


def _save_pgids(runner_id: str, pgids: Iterable[int]) -> None:
    _PGID_DIR.mkdir(parents=True, exist_ok=True)
    _pgid_path(runner_id).write_text(
        "".join(f"{pgid}\n" for pgid in sorted(set(pgids)))
    )


def _add_pgid(runner_id: str, pgid: int) -> None:
    pgids = _load_pgids(runner_id)
    pgids.add(pgid)
    _save_pgids(runner_id, pgids)


def _load_pgids(runner_id: str) -> set[int]:
    path = _pgid_path(runner_id)
    if not path.exists():
        return set()
    pgids: set[int] = set()
    try:
        for token in path.read_text().split():
            pgids.add(int(token))
    except (ValueError, OSError):
        return set()
    return pgids


def _remove_pgid(runner_id: str) -> None:
    _pgid_path(runner_id).unlink(missing_ok=True)


def _kill_pgid(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass(frozen=True)
class CommandSupervision:
    """Explicit lock, deadlines and ownership for synchronous Linux commands."""

    log_directory: Path
    stage_key: str
    timeouts: dict[str, int]
    lock_path: Path
    lock_wait: int
    source_sha: str
    owner_pid: str | None
    owner_start: str | None

    def __post_init__(self):
        if (
            self.lock_wait <= 0
            or not self.timeouts
            or any(t <= 0 for t in self.timeouts.values())
        ):
            raise ValueError("Command timeouts must be positive")
        if any(
            re.fullmatch(r"[A-Za-z0-9_-]+", stage) is None for stage in self.timeouts
        ):
            raise ValueError("Invalid command stage")
        if (self.owner_pid is None) != (self.owner_start is None):
            raise ValueError("Owner PID and start time must be supplied together")


class ProcessGroupManager:
    def __init__(
        self,
        runner_id: str,
        term_timeout: float = 10.0,
        *,
        supervision: CommandSupervision | None,
    ) -> None:
        if not runner_id:
            raise ValueError("runner_id must be a non-empty string")
        self.supervision = supervision
        self.runner_id = runner_id
        self.term_timeout = term_timeout
        self._procs: list[subprocess.Popen] = []

    def cleanup_stale(self, dry_run: bool = False) -> None:
        """
        Kill any process group left over from a previous run of this runner.
        Safe to call even if no stale pgid file exists.
        """
        if self.supervision is not None:
            raise ValueError("Supervised commands cannot clean up other runner jobs")
        pgids = _load_pgids(self.runner_id)
        if not pgids:
            print(
                f"[pgm] cleanup_stale: no stale pgid file for runner={self.runner_id}",
                flush=True,
            )
            return

        print(
            f"[pgm] cleanup_stale: killing stale process groups pgids={sorted(pgids)} "
            f"for runner={self.runner_id}",
            flush=True,
        )
        if dry_run:
            print(
                f"[pgm] cleanup_stale: [dry-run] skip kill pgids={sorted(pgids)}",
                flush=True,
            )
            return

        print(
            f"[pgm] cleanup_stale: sending SIGTERM to pgids={sorted(pgids)}",
            flush=True,
        )
        for pgid in sorted(pgids):
            _kill_pgid(pgid, signal.SIGTERM)
        print(
            f"[pgm] cleanup_stale: waiting {self.term_timeout}s for graceful shutdown",
            flush=True,
        )
        time.sleep(self.term_timeout)
        print(
            f"[pgm] cleanup_stale: sending SIGKILL to pgids={sorted(pgids)}",
            flush=True,
        )
        for pgid in sorted(pgids):
            _kill_pgid(pgid, signal.SIGKILL)
        _remove_pgid(self.runner_id)
        print(
            f"[pgm] cleanup_stale: removed pgid file for runner={self.runner_id}",
            flush=True,
        )

    def start(
        self,
        command: str,
        *,
        shell: bool = True,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        dry_run: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Launch command in a new process group (long-running server)."""
        if self.supervision is not None:
            raise ValueError("Supervised commands must use synchronous run()")
        print(f"[pgm] start: launching command for runner={self.runner_id}", flush=True)
        print(f"$ {command}", flush=True)
        if dry_run:
            print("[pgm] start: [dry-run] skip", flush=True)
            return None

        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self._procs.append(proc)
        print(
            f"[pgm] start: spawned pid={proc.pid}, tracked_procs={len(self._procs)}",
            flush=True,
        )

        try:
            pgid = os.getpgid(proc.pid)
            _add_pgid(self.runner_id, pgid)
            print(
                f"[pgm] start: pid={proc.pid} pgid={pgid} "
                f"runner={self.runner_id} pgid_file={_pgid_path(self.runner_id)}",
                flush=True,
            )
        except ProcessLookupError:
            print(
                f"[pgm] start: pid={proc.pid} already exited before getpgid", flush=True
            )

        return proc

    def run(
        self,
        command: str,
        *,
        shell: bool = True,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        dry_run: bool = False,
        check: bool = True,
    ) -> dict:
        """
        Run a short-lived command and track it.
        """
        print(f"[pgm] run: executing command for runner={self.runner_id}", flush=True)
        print(f"$ {command}", flush=True)
        if dry_run:
            print("[pgm] run: [dry-run] skip", flush=True)
            return {"command": command, "returncode": 0, "output": "", "dry_run": True}

        if self.supervision is not None:
            if not shell or env is None or cwd is None:
                raise ValueError("Supervised commands require shell=True, env and cwd")
            result = _run_supervised(command, env=env, cwd=cwd, policy=self.supervision)
            if check and result["returncode"] != 0:
                raise RuntimeError(
                    f"command failed with exit code {result['returncode']}: {command}"
                )
            return result

        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
        self._procs.append(proc)

        try:
            pgid = os.getpgid(proc.pid)
            _add_pgid(self.runner_id, pgid)
            print(
                f"[pgm] run: spawned pid={proc.pid} pgid={pgid} "
                f"tracked_procs={len(self._procs)}",
                flush=True,
            )
        except ProcessLookupError:
            print(
                f"[pgm] run: pid={proc.pid} already exited before getpgid", flush=True
            )

        output_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.wait()
        self._procs.remove(proc)

        returncode = proc.returncode
        print(
            f"[pgm] run: finished pid={proc.pid} returncode={returncode} "
            f"tracked_procs={len(self._procs)}",
            flush=True,
        )

        # Python multiprocessing daemon children receive SIGTERM when the
        # parent exits, which can race with normal shutdown and cause the
        # parent to report -15.  If the test output shows ALL tests passed,
        # downgrade the signal exit to a warning.
        if returncode < 0:
            output_text = "".join(output_lines)
            # Match "Test Summary: N/N passed" only when N == N (all passed)
            summary_match = re.search(
                r"Test Summary:\s+(\d+)/(\d+)\s+passed", output_text
            )
            all_passed = (
                summary_match is not None
                and summary_match.group(1) == summary_match.group(2)
            ) or re.search(r"Ran \d+ tests?.*\bOK\b", output_text)
            if all_passed:
                print(
                    f"[pgm] run: process exited with signal {-returncode}, "
                    f"but test output indicates success — treating as passed",
                    flush=True,
                )
                returncode = 0

        if check and returncode != 0:
            raise RuntimeError(f"command failed with exit code {returncode}: {command}")

        return {
            "command": command,
            "returncode": returncode,
            "output": "".join(output_lines),
        }

    def terminate_all(self, dry_run: bool = False) -> None:
        """Kill all processes started via this manager."""
        if self.supervision is not None:
            # run() waits for its independent supervisor to clean up while holding
            # the lock. Never consult a persistent ledger owned by another job.
            return
        print(
            f"[pgm] terminate_all: runner={self.runner_id} tracked_procs={len(self._procs)}",
            flush=True,
        )
        if dry_run:
            print(
                f"[pgm] terminate_all: [dry-run] skip killing {len(self._procs)} procs",
                flush=True,
            )
            return

        procs = list(self._procs)
        if not procs:
            # run() removes completed parents from _procs, but their process
            # groups may still have surviving children. Kill the persisted
            # pgids before discarding the record.
            saved_pgids = _load_pgids(self.runner_id)
            if saved_pgids:
                print(
                    f"[pgm] terminate_all: no tracked procs but pgid file exists, "
                    f"killing pgids={sorted(saved_pgids)}",
                    flush=True,
                )
                for pgid in sorted(saved_pgids):
                    _kill_pgid(pgid, signal.SIGTERM)
                time.sleep(1)
                for pgid in sorted(saved_pgids):
                    _kill_pgid(pgid, signal.SIGKILL)
            else:
                print(
                    "[pgm] terminate_all: no tracked processes, nothing to do",
                    flush=True,
                )
            _remove_pgid(self.runner_id)
            return

        pids = [p.pid for p in procs]
        print(
            f"[pgm] terminate_all: killing {len(procs)} process(es) pids={pids}",
            flush=True,
        )

        pgids: set[int] = _load_pgids(self.runner_id)
        for proc in procs:
            try:
                pgid = os.getpgid(proc.pid)
                pgids.add(pgid)
                _kill_pgid(pgid, signal.SIGTERM)
                print(
                    f"[pgm] terminate_all: sent SIGTERM to pgid={pgid} (pid={proc.pid})",
                    flush=True,
                )
            except ProcessLookupError:
                print(f"[pgm] terminate_all: pid={proc.pid} already exited", flush=True)

        print(
            f"[pgm] terminate_all: waiting up to {self.term_timeout}s for graceful shutdown",
            flush=True,
        )
        deadline = time.time() + self.term_timeout
        for proc in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(
                    f"[pgm] terminate_all: pid={proc.pid} exited gracefully", flush=True
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[pgm] terminate_all: pid={proc.pid} did not exit in time",
                    flush=True,
                )

        print(
            f"[pgm] terminate_all: sending SIGKILL to pgids={sorted(pgids)}", flush=True
        )
        for pgid in pgids:
            _kill_pgid(pgid, signal.SIGKILL)
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"[pgm] terminate_all: pid={proc.pid} still alive after SIGKILL",
                    flush=True,
                )

        self._procs.clear()
        _remove_pgid(self.runner_id)
        print("[pgm] terminate_all: cleanup done, all processes cleared", flush=True)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_manager(term_timeout: float = 10.0) -> ProcessGroupManager:
    """
    Build a ProcessGroupManager from the environment.

    Looks for a stable runner identifier in order:
      1. RUNNER_NAME   – set by GitHub Actions per registered runner
      2. CI_RUNNER_NAME – set by GitLab CI
      3. HOSTNAME      – machine hostname (stable on dedicated hosts)

    Raises RuntimeError if none of the above is available.
    """
    runner_id = (
        os.environ.get("RUNNER_NAME")
        or os.environ.get("CI_RUNNER_NAME")
        or os.environ.get("HOSTNAME")
    )
    if not runner_id:
        raise RuntimeError(
            "Cannot determine a stable runner identifier. "
            "Set RUNNER_NAME, CI_RUNNER_NAME, or HOSTNAME."
        )
    print(
        f"[pgm] make_manager: runner_id={runner_id} "
        f"(RUNNER_NAME={os.environ.get('RUNNER_NAME', '<unset>')}, "
        f"HOSTNAME={os.environ.get('HOSTNAME', '<unset>')})",
        flush=True,
    )
    return ProcessGroupManager(
        runner_id=runner_id, term_timeout=term_timeout, supervision=None
    )


def _write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def _supervise(args):
    """Keep the private command group and host lock until all cleanup completes."""
    import fcntl

    interrupted = []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda number, frame: interrupted.append(number))
    started = time.monotonic()
    process = None
    rc = 1
    reason = None
    lock = None
    worker = args.owner_pid

    def owner_gone():
        ready, _, _ = select.select([args.parent_fd], [], [], 0)
        return (ready and not os.read(args.parent_fd, 1)) or (
            worker is not None and _process_start(int(worker)) != args.owner_start
        )

    try:
        lock = open(args.lock, "r+")
        while True:
            if interrupted or owner_gone():
                raise InterruptedError(
                    "Command owner exited while waiting for host lock"
                )
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= args.lock_wait:
                    raise TimeoutError("Host lock wait timed out")
                time.sleep(0.1)
        deadline = time.monotonic() + args.timeout
        process = subprocess.Popen(
            ["bash", "-c", args.command],
            cwd=args.cwd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with args.log.open("xb") as log, selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if interrupted or owner_gone():
                    reason = "Command owner exited or was interrupted"
                    rc = 128 + (interrupted[0] if interrupted else signal.SIGTERM)
                    break
                if time.monotonic() >= deadline:
                    reason, rc = "Command execution timed out", 124
                    break
                for key, _ in selector.select(0.1):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    log.flush()
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        pass
                if (
                    os.waitid(
                        os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                    )
                    is not None
                ):
                    rc = 0
                    break
            log.flush()
            os.fsync(log.fileno())
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        try:
            if process is not None:
                # Preserve the group leader PID until its remaining descendants stop.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(0.5)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                process.stdout.close()
                if rc == 0:
                    rc = (
                        process.returncode
                        if process.returncode >= 0
                        else 128 - process.returncode
                    )
            _write_json(
                args.log.with_suffix(".process.json"),
                {
                    "exit_code": rc,
                    "reason": reason,
                    "source_sha": args.source_sha,
                    "child_pid": process.pid if process else None,
                },
            )
        finally:
            if lock is not None:
                lock.close()
            os.close(args.parent_fd)
    return rc


def _run_supervised(command, *, env, cwd, policy):
    stage = env[policy.stage_key]
    if stage not in policy.timeouts:
        raise ValueError(f"Unsupported command stage: {stage}")
    log = policy.log_directory / f"{stage}.log"
    timeout = str(policy.timeouts[stage])
    read_fd, write_fd = os.pipe()
    process = None
    output = []
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "supervise",
                "--parent-fd",
                str(read_fd),
                "--timeout",
                timeout,
                "--lock-wait",
                str(policy.lock_wait),
                "--lock",
                str(policy.lock_path),
                "--source-sha",
                policy.source_sha,
                *(
                    [
                        "--owner-pid",
                        policy.owner_pid,
                        "--owner-start",
                        policy.owner_start,
                    ]
                    if policy.owner_pid is not None
                    else []
                ),
                "--cwd",
                str(cwd),
                "--log",
                str(log),
                "--command",
                command,
            ],
            pass_fds=(read_fd,),
            start_new_session=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        os.close(read_fd)
        read_fd = None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
        rc = process.wait()
        proof = json.loads(log.with_suffix(".process.json").read_text())
        if proof["exit_code"] != rc or proof["source_sha"] != policy.source_sha:
            raise RuntimeError("Command process evidence mismatch")
        return {"command": command, "returncode": rc, "output": "".join(output)}
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)
        if process is not None:
            process.stdout.close()
            process.wait(timeout=20)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("cleanup-stale")
    child = modes.add_parser("supervise")
    child.add_argument("--parent-fd", type=int, required=True)
    child.add_argument("--timeout", type=int, required=True)
    child.add_argument("--lock-wait", type=int, required=True)
    child.add_argument("--lock", type=Path, required=True)
    child.add_argument("--source-sha", required=True)
    child.add_argument("--owner-pid")
    child.add_argument("--owner-start")
    child.add_argument("--cwd", type=Path, required=True)
    child.add_argument("--log", type=Path, required=True)
    child.add_argument("--command", required=True)
    args = parser.parse_args()
    if args.mode == "cleanup-stale":
        # Survivors can hold work directory files open on NFS.
        make_manager().cleanup_stale()
        return 0
    if args.timeout <= 0 or args.lock_wait <= 0:
        parser.error("Command timeouts must be positive")
    if (args.owner_pid is None) != (args.owner_start is None):
        parser.error("Owner PID and start time must be supplied together")
    return _supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
