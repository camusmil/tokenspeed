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

"""Run one registered NPU test phase and preserve its evidence."""

from __future__ import annotations

import argparse
import codecs
import importlib.metadata
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import ci_reporting
from ci_test_plan import file_digest, freeze_plan, read_frozen, verify_identity
from task_config import CONFIG_PATH, load_task


def environment_packages():
    versions = {}
    for name in ("torch", "torch-npu", "triton-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def positive_seconds(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("accuracy", "performance"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=positive_seconds, required=True)
    parser.add_argument(
        "--github-summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or None,
    )
    return parser.parse_args(argv)


def load_inputs(repo_root, phase, work):
    root = Path(repo_root).resolve(strict=True)
    task = load_task(root / CONFIG_PATH)
    reporting = vars(ci_reporting)
    if phase == "accuracy":
        plan, plan_digest = freeze_plan(root, task, work, capture_test, save_text)
    else:
        report = ci_reporting.decode_record(
            (Path(work) / "accuracy/report.json").read_text()
        )
        ci_reporting.require(
            report["exit_code"] == 0, "Accuracy did not complete successfully"
        )
        plan_digest = report["metadata"]["plan_sha256"]
        plan = read_frozen(root, task, work, plan_digest)
    return root, task, plan, plan_digest, reporting


def stop_process_group(process):
    """Stop the private test process group, including surviving children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def capture_test(command, repo_root, log_path, timeout_seconds):
    """Stream merged output to disk and console; preserve the child exit code."""
    process = None
    parent_write = None
    parent_read = None
    saved = False
    capture_error = None
    path = Path(log_path)
    try:
        with path.open("xb") as log:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            started = time.monotonic()
            deadline = started + timeout_seconds
            next_heartbeat = started + 30
            try:
                passed_fds = ()
                if sys.platform == "linux":
                    parent_read, parent_write = os.pipe()
                    passed_fds = (parent_read,)
                    try:
                        os.fstat(9)
                        passed_fds += (9,)
                    except OSError:
                        pass
                    command = [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("ci_lifecycle.py")),
                        "supervise",
                        str(parent_read),
                        *command,
                    ]
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    bufsize=0,
                    pass_fds=passed_fds,
                    env=dict(
                        os.environ,
                        NPU_PROCESS_RECORD=str(path.with_suffix(".owner.json")),
                    ),
                )
                if parent_read is not None:
                    os.close(parent_read)
                    parent_read = None
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        now = time.monotonic()
                        if now >= next_heartbeat:
                            print(
                                f"Test is still running: elapsed={now-started:.1f}s, log={path}",
                                flush=True,
                            )
                            next_heartbeat = now + 30
                        remaining = deadline - now
                        if remaining <= 0:
                            raise TimeoutError(
                                f"Test exceeded {timeout_seconds} seconds"
                            )
                        for key, _ in selector.select(min(remaining, 0.2)):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if log.write(chunk) != len(chunk):
                                raise OSError("Incomplete test log write")
                            log.flush()
                            sys.stdout.write(decoder.decode(chunk))
                            sys.stdout.flush()
                sys.stdout.write(decoder.decode(b"", final=True))
                sys.stdout.flush()
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process is not None:
                    try:
                        stop_process_group(process)
                    finally:
                        process.stdout.close()
                for fd in (parent_read, parent_write):
                    if fd is not None:
                        os.close(fd)
            log.flush()
            os.fsync(log.fileno())
        saved = True
    except Exception as exc:
        capture_error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "exit_code": None if process is None else process.returncode,
        "log_path": str(path),
        "log_saved": saved,
        "capture_error": capture_error,
    }


def save_text(path, text):
    raw = text.encode("utf-8")
    with Path(path).open("xb") as handle:
        if handle.write(raw) != len(raw):
            raise OSError(f"Incomplete evidence write: {path}")
        handle.flush()
        os.fsync(handle.fileno())
    if Path(path).read_bytes() != raw:
        raise OSError(f"Saved evidence differs: {path}")


def run_operator(root, entry, phase, plan, reporting, output_dir, timeout):
    name = entry["name"]
    directory = Path(output_dir) / name
    log = directory / (phase + ".log")
    result = {
        "operator": name,
        "phase": phase,
        "status": "failed",
        "exit_code": None,
        "log_path": str(log),
        "structured_log_valid": False,
        "log_saved": False,
    }
    try:
        directory.mkdir()
        command = [
            sys.executable,
            "-B",
            "-u",
            str(root / entry["script"]),
            "--mode",
            phase,
        ]
        captured = capture_test(command, root, log, timeout)
        result.update(exit_code=captured["exit_code"], log_saved=captured["log_saved"])
        timing = plan["timing"] if phase == "performance" else {}
        result = reporting["summarize_phase"](
            log,
            name,
            phase,
            plan[phase],
            captured["exit_code"],
            **timing,
        )
        result["log_saved"] = captured["log_saved"]
        result["capture_error"] = captured["capture_error"]
        if captured["capture_error"] is not None:
            result["status"] = "failed"
            result["validation_error"] = result.get("error")
            result["error_type"] = captured["capture_error"]["type"]
            result["error"] = captured["capture_error"]["message"]
        save_text(
            directory / "process.json",
            json.dumps(captured, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        if captured["exit_code"] is not None:
            save_text(directory / (phase + ".rc"), str(captured["exit_code"]) + "\n")
        save_text(
            directory / "result.json",
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
    except Exception as exc:
        result["status"] = "failed"
        result["log_saved"] = False
        result["evidence_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result.setdefault("error_type", type(exc).__name__)
        previous = result.get("error")
        message = f"Operator execution/evidence error: {type(exc).__name__}: {exc}"
        result["error"] = f"{previous}; {message}" if previous else message
    return result


def execute_phase(
    root,
    task,
    plan,
    plan_digest,
    reporting,
    phase,
    output_dir,
    timeout,
    github_summary=None,
):
    reporting["require"](phase in task["execution"]["phases"], "Unknown phase")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "repo_root": str(root),
        "python": sys.executable,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_sha256": plan["task_sha256"],
        "plan_sha256": plan_digest,
        "source_sha": plan["source_sha"],
        "environment": {
            "python_version": sys.version,
            "packages": environment_packages(),
            "device_ids": task["resources"]["device_ids"],
            "environment_entry": os.environ.get("NPU_ENVIRONMENT_ENTRY"),
            "environment_sha256": os.environ.get("NPU_ENVIRONMENT_SHA256"),
        },
        "timeout_seconds": timeout,
        "lock_wait_seconds": os.environ.get("NPU_LOCK_WAIT_SECONDS"),
        "evidence_directory": str(output),
    }
    save_text(
        output / "context.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    results = []
    actions = os.environ.get("GITHUB_ACTIONS") == "true"
    deadline = time.monotonic() + timeout
    for entry in task["operators"]:
        verify_identity(root, task, plan)
        ci_reporting.require(time.monotonic() < deadline, "Phase execution timeout")
        name = entry["name"]
        started = time.monotonic()
        print(f"Execution started: {name}/{phase}", flush=True)
        if actions:
            print(f"::group::{name}/{phase}", flush=True)
        try:
            result = run_operator(
                root,
                entry,
                phase,
                plan["operators"][name],
                reporting,
                output,
                max(0.001, deadline - time.monotonic()),
            )
        finally:
            if actions:
                print("::endgroup::", flush=True)
        print(
            f"Execution finished: {name}/{phase}, exit={result['exit_code']}, "
            f"elapsed={time.monotonic()-started:.2f}s, status={result['status']}",
            flush=True,
        )
        verify_identity(root, task, plan)
        ci_reporting.require(
            file_digest(output.parent / "frozen-plan.json") == plan_digest,
            "Frozen plan changed during execution",
        )
        results.append(result)
        if not reporting["phase_succeeded"](result, phase):
            detail = result.get("error") or f"process exit={result['exit_code']}"
            report_error(f"{name}/{phase}: {detail}")
            print(
                f"{name}/{phase} failed; remaining operators will not run.", flush=True
            )
            break
    metadata["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = reporting["make_report"](results, metadata, task, phase)
    print(reporting["render_summary"](report), flush=True)
    saved = reporting["save_report"](report, output, github_summary)
    print("Structured report:", saved["json"]["path"], flush=True)
    print("Markdown summary:", saved["markdown"]["path"], flush=True)
    return report["exit_code"]


def report_error(message):
    print(message, file=sys.stderr, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}", flush=True)


def save_failure_summary(path, phase, message):
    if path is not None:
        # Escape raw errors rather than interpreting them as Markdown/HTML.
        import html

        text = (
            f"\n### ARM NPU phase execution failed\n\nPhase: `{phase}`. Remaining tests have been stopped.\n\n"
            f"<pre>{html.escape(message)}</pre>\n"
        )
        with Path(path).open("a", encoding="utf-8") as handle:
            if handle.write(text) != len(text):
                raise OSError("Incomplete failure summary write")


class PhaseInterrupted(BaseException):
    def __init__(self, signum):
        super().__init__(f"Interrupted by signal {signum}")
        self.signum = signum


def handle_interrupt(signum, frame):
    # Allow the active capture's finally block to finish process cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise PhaseInterrupted(signum)


def main(argv=None):
    args = parse_args(argv)
    previous_handlers = {}
    previous_pythonpath = os.environ.get("PYTHONPATH")
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_interrupt)
        root, task, plan, plan_digest, reporting = load_inputs(
            args.repo_root, args.phase, args.output_dir.parent
        )
        paths = [
            root / "tokenspeed-kernel/python",
            root / "tokenspeed-kernel-npu/python",
        ]
        for path in paths:
            if not path.is_dir():
                raise FileNotFoundError(f"Checkout source directory missing: {path}")
        pythonpath = os.pathsep.join(str(path) for path in paths)
        if previous_pythonpath:
            pythonpath += os.pathsep + previous_pythonpath
        os.environ["PYTHONPATH"] = pythonpath
        if os.environ.get("GITHUB_ACTIONS") == "true" and args.github_summary is None:
            raise RuntimeError("GITHUB_STEP_SUMMARY is required inside Actions")
        print("Phase:", args.phase, flush=True)
        print("Python:", sys.executable, flush=True)
        print("Source directory:", root, flush=True)
        return execute_phase(
            root,
            task,
            plan,
            plan_digest,
            reporting,
            args.phase,
            args.output_dir,
            args.timeout_seconds,
            args.github_summary,
        )
    except PhaseInterrupted as exc:
        message = f"Phase interrupted: signal={exc.signum}; remaining tests have been stopped."
        report_error(message)
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_text(Path(args.output_dir) / "executor-error.log", message + "\n")
            save_failure_summary(args.github_summary, args.phase, message)
        except Exception as summary_error:
            report_error(f"Failed to save summary: {summary_error}")
        return 128 + exc.signum
    except Exception as exc:
        message = f"Phase execution failed: {type(exc).__name__}: {exc}"
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        report_error(message)
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_text(Path(args.output_dir) / "executor-error.log", detail)
            save_failure_summary(args.github_summary, args.phase, message)
        except Exception as summary_error:
            report_error(f"Failed to save summary: {summary_error}")
        return 1
    finally:
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
