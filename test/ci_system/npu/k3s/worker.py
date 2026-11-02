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


"""Registered four-stage native NPU chain. No network or management credentials."""

import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

STAGES = [
    "operator-accuracy",
    "operator-performance",
    "model-accuracy",
    "model-performance",
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(command, *, cwd, env, log, timeout):
    with log.open("w") as f:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return child.wait(timeout=timeout)
        finally:
            # Includes surviving children of an exited command; never scans host processes.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            until = time.monotonic() + 5
            while time.monotonic() < until:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def archive(results):
    files = {}
    for p in sorted(results.rglob("*")):
        if p.is_symlink():
            raise ValueError("symlink in evidence")
        if p.is_file():
            files[p.relative_to(results).as_posix()] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    save(results / "files.json", files)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(results.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(results).as_posix(), recursive=False)
    body = buf.getvalue()
    if len(body) > 8 * 1024 * 1024:
        raise ValueError("evidence exceeds registered 8MiB limit")
    print(
        "CP12_ARCHIVE_BEGIN "
        + json.dumps({"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}),
        flush=True,
    )
    encoded = base64.b64encode(body).decode()
    for offset in range(0, len(encoded), 32768):
        print("CP12_ARCHIVE " + encoded[offset : offset + 32768], flush=True)
    print("CP12_ARCHIVE_END", flush=True)


def execute(args):
    work = Path("/work")
    results = work / "results"
    results.mkdir(exist_ok=False)
    root = work / "checkout"
    env = dict(os.environ)
    forbidden = [
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "KUBECONFIG",
        "HF_TOKEN",
    ]
    if (
        any(k in env for k in forbidden)
        or Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()
    ):
        raise ValueError("management credentials must be absent")
    summary = {
        "schema_version": 1,
        "task_id": args.task,
        "source_sha": args.sha,
        "scenario": args.scenario,
        "stages": [],
        "quality_verdict": "not_evaluated",
        "performance_verdict": "not_evaluated",
        "ok": False,
        "started_at": time.time(),
        "management_credentials_absent": True,
    }
    lock = open("/host-lock/host-all.lock", "rb")
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("host lock wait exceeded")
                time.sleep(1)
        summary["lock_inode"] = os.fstat(lock.fileno()).st_ino
        shutil.copytree("/source", root, symlinks=True)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if actual != args.sha:
            raise ValueError("source SHA mismatch")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
        ).strip():
            raise ValueError("dirty source")
        # No editable reference to a previous job. All build/cache paths are task-local.
        env.update(
            NPU_CI_WORK=str(work),
            NPU_CI_SOURCE_ROOT=str(root),
            TOKENSPEED_CI_SOURCE_SHA=args.sha,
            NPU_CI_BOOTSTRAP_PYTHON=sys.executable,
            NPU_CI_WHEELS="/deps/runtime-wheels-20261102",
            NPU_CI_ENGINE_DEPS="/deps/engine-build-deps-20261026",
            NPU_CI_OPENSSL_ROOT="/deps/openssl-3.0.8",
            NPU_CI_PYTHON_LIBDIR="/srv/tokenspeed-npu/cann9-torch29-py312-root/python/lib",
            TOKENSPEED_KERNEL_BACKEND="npu",
            TOKENSPEED_KERNEL_GIT_SHA=args.sha,
            TOKENSPEED_KERNEL_GIT_BRANCH="ci/npu",
            TOKENSPEED_KERNEL_NIGHTLY="false",
            PIP_CONFIG_FILE="/dev/null",
            PIP_NO_INDEX="1",
            PIP_FIND_LINKS="/deps/runtime-wheels-20261102",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            GITHUB_ACTIONS="true",
            GITHUB_STEP_SUMMARY=str(results / "summary.md"),
            LD_LIBRARY_PATH="/deps/openssl-3.0.8/lib64:/deps/openssl-3.0.8/lib:"
            + env.get("LD_LIBRARY_PATH", ""),
            CMAKE_ARGS="-DOPENSSL_ROOT_DIR=/deps/openssl-3.0.8",
            CMAKE_BUILD_PARALLEL_LEVEL="4",
            MAX_JOBS="4",
        )
        for key, relative in {
            "TMPDIR": "tmp",
            "XDG_CACHE_HOME": "cache",
            "TRITON_CACHE_DIR": "cache/triton",
            "TVM_FFI_CACHE_DIR": "cache/tvm",
            "TORCH_EXTENSIONS_DIR": "cache/torch",
            "HF_HOME": "cache/hf",
        }.items():
            p = work / relative
            p.mkdir(parents=True, exist_ok=True)
            env[key] = str(p)
        (work / "runtime").mkdir()
        (results / "summary.md").touch()
        code = root / "test/ci_system/npu/k3s"
        summary["environment"] = {
            "python": sys.executable,
            "base_image": args.image,
            "source_sha": actual,
        }
        summary["manifest_sha256"] = hashlib.sha256(
            (root / "test/ci/npu/manifest.json").read_bytes()
        ).hexdigest()
        print(
            "CP12_STARTED "
            + json.dumps(
                {
                    "task_id": args.task,
                    "source_sha": args.sha,
                    "lock_inode": summary["lock_inode"],
                }
            ),
            flush=True,
        )
        for stage in STAGES:
            item = {"name": stage, "started_at": time.time(), "status": "failed"}
            summary["stages"].append(item)
            print("CP12_STAGE " + json.dumps(item), flush=True)
            if stage.startswith("operator-"):
                phase = stage.split("-")[1]
                cmd = [
                    sys.executable,
                    "-B",
                    "-u",
                    str(root / "test/ci_system/npu/run_phase.py"),
                    "--repo-root",
                    str(root),
                    "--phase",
                    phase,
                    "--output-dir",
                    str(results / stage),
                    "--timeout-seconds",
                    "600",
                ]
                rc = run(
                    cmd, cwd=root, env=env, log=results / (stage + ".log"), timeout=900
                )
                item["exit_code"] = rc
                if rc:
                    raise RuntimeError(f"{stage} failed: {rc}")
                if args.scenario == "operator-failure" and stage == "operator-accuracy":
                    item["injected_failure"] = "registered post-accuracy failure"
                    raise RuntimeError("injected operator accuracy failure")
            else:
                if not (work / "runtime/install-evidence.json").exists():
                    for name in ["install_runtime.py", "prove_runtime.py"]:
                        interpreter = (
                            str(work / "model-venv/bin/python")
                            if name == "prove_runtime.py"
                            else sys.executable
                        )
                        rc = run(
                            [interpreter, "-B", "-u", str(code / name)],
                            cwd=root,
                            env=env,
                            log=results / (name + ".log"),
                            timeout=1800,
                        )
                        if rc:
                            raise RuntimeError(f"{name} failed: {rc}")
                    shutil.copytree(
                        work / "runtime", results / "runtime", dirs_exist_ok=True
                    )
                python = str(work / "model-venv/bin/python")
                out = results / stage
                out.mkdir()
                model = Path(
                    "/deps/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca"
                )
                model_manifest = model / "model-manifest.json"
                config = json.loads((code / "model-profile.json").read_text())
                cmd = [
                    python,
                    "-B",
                    "-u",
                    str(code / "model_phase.py"),
                    "--stage",
                    stage,
                    "--scenario",
                    args.scenario,
                    "--root",
                    str(root),
                    "--output",
                    str(out),
                    "--sha",
                    args.sha,
                    "--model",
                    str(model),
                    "--manifest-sha",
                    hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
                ]
                rc = run(
                    cmd, cwd=root, env=env, log=results / (stage + ".log"), timeout=900
                )
                item["exit_code"] = rc
                if rc:
                    raise RuntimeError(f"{stage} failed: {rc}")
            item.update(status="passed", finished_at=time.time())
            save(results / "chain.json", summary)
        summary["ok"] = True
    except BaseException as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        if summary["stages"]:
            summary["stages"][-1]["finished_at"] = time.time()
        # Copy diagnostic install inputs even when the build failed.
        if (work / "runtime").exists():
            shutil.copytree(work / "runtime", results / "runtime", dirs_exist_ok=True)
    finally:
        summary["finished_at"] = time.time()
        save(results / "chain.json", summary)
        try:
            archive(results)
        finally:
            lock.close()
    print("CP12_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
    return 0 if summary["ok"] else 42


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument(
        "--scenario",
        required=True,
        choices=["normal", "operator-failure", "model-failure", "save-failure"],
    )
    p.add_argument("--image", required=True)

    def interrupted(signum, frame):
        raise InterruptedError(f"worker interrupted: {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(execute(p.parse_args()))
