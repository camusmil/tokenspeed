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

"""NPU environment preparation used by the public YAML pipeline.

Task discovery, stage ordering and execution remain in pipeline.py. This
adapter provides a job-local native environment and bounded process ownership.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import pwd
import select
import selectors
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

PROJECTS = {
    "tokenspeed": "python",
    "tokenspeed-scheduler": "tokenspeed-scheduler",
    "tokenspeed-kernel": "tokenspeed-kernel/python",
    "tokenspeed-kernel-npu": "tokenspeed-kernel-npu",
}
BASE_REQUIREMENTS = [
    "torch==2.9.0+cpu",
    "torch-npu==2.9.0.post2",
    "triton-ascend==3.2.1",
    "numpy==1.26.4",
    "transformers==5.12.0",
    "apache-tvm-ffi==0.1.13.post3",
]


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def runtime_environment(env, root, *, dry_run):
    if dry_run:
        return dict(env)
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("NPU runtime CI requires Linux ARM64")
    if pwd.getpwuid(os.geteuid()).pw_name != "ci-npu":
        raise RuntimeError("NPU runtime CI requires the ci-npu account")
    root = root.resolve(strict=True)
    work = Path(env["NPU_CI_WORK"]).resolve(strict=True)
    if root != work / "checkout":
        raise ValueError("Runtime source must be this task's checkout")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if sha != env["GITHUB_SHA"]:
        raise ValueError("Runtime source differs from the event commit")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True)
    config = Path(env["NPU_CI_DEPLOYMENT"])
    if (
        config.is_symlink()
        or config.stat().st_uid != 0
        or config.stat().st_mode & 0o022
    ):
        raise ValueError("NPU deployment must be a root-owned protected regular file")
    deployment = json.loads(config.read_text())
    for key in ("environment", "python", "wheels", "openssl_root", "model"):
        value = Path(deployment[key])
        if not value.is_absolute() or not value.exists():
            raise ValueError(f"Missing NPU deployment path: {key}")
    raw = subprocess.check_output(
        [
            "bash",
            "-c",
            'set -e; source "$1" >/dev/null; source "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh" >/dev/null; env -0',
            "npu-runtime-env",
            deployment["environment"],
        ],
        env=env,
    )
    result = dict(env)
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = os.fsdecode(item).split("=", 1)
            result[key] = value
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "TORCH_TRANSFER_TO_NPU",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_CONSTRAINT",
        "PIP_NO_INDEX",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
    ):
        result.pop(name, None)
    venv = work / "model-venv"
    result.update(
        NPU_CI_SOURCE_ROOT=str(root),
        TOKENSPEED_CI_SOURCE_SHA=sha,
        NPU_CI_BOOTSTRAP_PYTHON=deployment["python"],
        NPU_CI_WHEELS=deployment["wheels"],
        NPU_SMOKE_MODEL_PATH=deployment["model"],
        VIRTUAL_ENV=str(venv),
        PATH=f"{venv / 'bin'}{os.pathsep}{result['PATH']}",
        PIP_CONFIG_FILE="/dev/null",
        PIP_REQUIRE_VIRTUALENV="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        ASCEND_RT_VISIBLE_DEVICES="0",
        TOKENSPEED_KERNEL_BACKEND="npu",
        TOKENSPEED_KERNEL_GIT_SHA=sha,
        TOKENSPEED_KERNEL_GIT_BRANCH="ci/npu",
        TOKENSPEED_KERNEL_NIGHTLY="false",
        CMAKE_ARGS=f"-DOPENSSL_ROOT_DIR={deployment['openssl_root']}",
        CMAKE_BUILD_PARALLEL_LEVEL="4",
    )
    for name, relative in {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "cache",
        "TRITON_CACHE_DIR": "cache/triton",
        "TVM_FFI_CACHE_DIR": "cache/tvm",
        "TORCH_EXTENSIONS_DIR": "cache/torch",
        "TORCH_HOME": "cache/torch-home",
        "HF_HOME": "cache/huggingface",
        "MODELSCOPE_CACHE": "cache/modelscope",
    }.items():
        path = work / relative
        path.mkdir(parents=True, exist_ok=True)
        result[name] = str(path)
    (work / "runtime").mkdir(exist_ok=True)
    parent = os.getpid()
    while parent > 1:
        stat = Path(f"/proc/{parent}/stat").read_text().rsplit(") ", 1)[1].split()
        if Path(f"/proc/{parent}/comm").read_text().strip() == "Runner.Worker":
            result.update(NPU_CI_WORKER_PID=str(parent), NPU_CI_WORKER_START=stat[19])
            break
        parent = int(stat[1])
    return result


def install_runtime():
    root = Path(os.environ["NPU_CI_SOURCE_ROOT"])
    work = Path(os.environ["NPU_CI_WORK"])
    venv = work / "model-venv"
    python = str(venv / "bin/python")
    subprocess.run(
        [os.environ["NPU_CI_BOOTSTRAP_PYTHON"], "-m", "venv", str(venv)], check=True
    )
    constraints = work / "runtime/constraints.txt"
    constraints.write_text("\n".join(BASE_REQUIREMENTS + ["triton==3.5.0"]) + "\n")
    os.environ["PIP_CONSTRAINT"] = str(constraints)
    wheels = os.environ["NPU_CI_WHEELS"]
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            wheels,
            *BASE_REQUIREMENTS,
        ],
        check=True,
    )
    requirements = set()
    for relative in PROJECTS.values():
        with (root / relative / "pyproject.toml").open("rb") as handle:
            requirements.update(tomllib.load(handle)["build-system"]["requires"])
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--timeout",
            "15",
            "--retries",
            "1",
            "--index-url",
            "https://pypi.org/simple",
            *sorted(requirements),
        ],
        check=True,
    )
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--timeout",
            "15",
            "--retries",
            "1",
            "--index-url",
            "https://pypi.org/simple",
            "--find-links",
            wheels,
            "--report",
            str(work / "runtime/pip-report.json"),
            *[str(root / p) for p in PROJECTS.values()],
        ],
        check=True,
    )
    subprocess.run([python, "-B", str(Path(__file__).resolve()), "proof"], check=True)


def record_install():
    root = Path(os.environ["NPU_CI_SOURCE_ROOT"])
    work = Path(os.environ["NPU_CI_WORK"])
    if Path(sys.prefix).resolve() != (work / "model-venv").resolve():
        raise ValueError("Installation proof requires this task's virtualenv")
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    (work / "runtime/pip-check.log").write_text(check.stdout + check.stderr)
    check.check_returncode()
    installed = {}
    for name, relative in PROJECTS.items():
        distribution = importlib.metadata.distribution(name)
        direct = json.loads(distribution.read_text("direct_url.json"))
        if direct["url"] != (root / relative).as_uri():
            raise ValueError(
                f"Runtime component is not installed from this checkout: {name}"
            )
        item = {"version": distribution.version, "direct_url": direct}
        if (
            name == "tokenspeed-kernel"
            and f"git{os.environ['TOKENSPEED_CI_SOURCE_SHA'][:8]}"
            not in distribution.version
        ):
            raise ValueError("Installed kernel version has a different source SHA")
        if name == "tokenspeed-scheduler":
            import tokenspeed_scheduler.tokenspeed_scheduler_ext as extension

            binary = Path(extension.__file__).resolve(strict=True)
            if not binary.is_relative_to(Path(sys.prefix).resolve()):
                raise ValueError(
                    "Scheduler extension is outside this task's virtualenv"
                )
            libraries = subprocess.check_output(["ldd", str(binary)], text=True)
            if "not found" in libraries:
                raise ValueError("Scheduler extension has missing libraries")
            item["binary"] = {
                "path": str(binary),
                "sha256": digest(binary),
                "ldd": libraries,
            }
        installed[name] = item
    write_json(
        work / "runtime/install-evidence.json",
        {
            "schema_version": 1,
            "components": "full-runtime",
            "source_sha": os.environ["TOKENSPEED_CI_SOURCE_SHA"],
            "source_root": str(root),
            "pip_check_passed": True,
            "installed": installed,
            "pip_report_sha256": digest(work / "runtime/pip-report.json"),
        },
    )


def supervise(args):
    """Keep the private command group and host lock until all cleanup completes."""
    interrupted = []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda number, frame: interrupted.append(number))
    started = time.monotonic()
    process = None
    rc = 1
    reason = None
    lock = None
    worker = os.environ.get("NPU_CI_WORKER_PID")

    def owner_gone():
        ready, _, _ = select.select([args.parent_fd], [], [], 0)
        return (ready and not os.read(args.parent_fd, 1)) or (
            worker is not None
            and process_start(int(worker)) != os.environ["NPU_CI_WORKER_START"]
        )

    try:
        lock = open(os.environ["NPU_CI_LOCK"], "r+")
        while True:
            if interrupted or owner_gone():
                raise InterruptedError(
                    "NPU command owner exited while waiting for host lock"
                )
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= args.lock_wait:
                    raise TimeoutError("NPU host lock wait timed out")
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
                    reason = "NPU command owner exited or was interrupted"
                    rc = 128 + (interrupted[0] if interrupted else signal.SIGTERM)
                    break
                if time.monotonic() >= deadline:
                    reason, rc = "NPU command execution timed out", 124
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
        write_json(
            args.log.with_suffix(".process.json"),
            {
                "exit_code": rc,
                "reason": reason,
                "source_sha": os.environ.get("TOKENSPEED_CI_SOURCE_SHA"),
                "child_pid": process.pid if process else None,
            },
        )
        if lock is not None:
            lock.close()
        os.close(args.parent_fd)
    return rc


def run_shell(command, *, env, cwd):
    stage = env["NPU_CI_STAGE"]
    if stage not in ("install", "ut"):
        raise ValueError(f"Unsupported NPU runtime stage: {stage}")
    log = Path(env["NPU_CI_WORK"]) / "runtime" / f"{stage}.log"
    timeout = env[
        "NPU_CI_INSTALL_TIMEOUT" if stage == "install" else "NPU_CI_TEST_TIMEOUT"
    ]
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
                env["NPU_CI_LOCK_TIMEOUT"],
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
        if (
            proof["exit_code"] != rc
            or proof["source_sha"] != env["TOKENSPEED_CI_SOURCE_SHA"]
        ):
            raise RuntimeError("NPU command process evidence mismatch")
        return {"command": command, "returncode": rc, "output": "".join(output)}
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)
        if process is not None:
            process.stdout.close()
            process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("install")
    subparsers.add_parser("proof")
    child = subparsers.add_parser("supervise")
    child.add_argument("--parent-fd", type=int, required=True)
    child.add_argument("--timeout", type=int, required=True)
    child.add_argument("--lock-wait", type=int, required=True)
    child.add_argument("--cwd", type=Path, required=True)
    child.add_argument("--log", type=Path, required=True)
    child.add_argument("--command", required=True)
    args = parser.parse_args()
    if args.mode == "install":
        install_runtime()
        return 0
    if args.mode == "proof":
        record_install()
        return 0
    if args.timeout <= 0 or args.lock_wait <= 0:
        parser.error("NPU timeouts must be positive")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
