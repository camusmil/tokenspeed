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

"""CPU tests for the public NPU runtime registration and execution contract."""

import copy
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from ci_register import DeviceBackend, collect_tests
from npu_runtime import run_shell
from pipeline import build_matrix, normalize_task, summarize_task_targets

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test/runtime/models/test_qwen3_npu_smoke.py"


def test_npu_runtime_is_registered_separately_from_cuda():
    entries = collect_tests([str(SCRIPT)], sanity_check=True)
    assert len(entries) == 1
    assert entries[0].backend == DeviceBackend.NPU
    assert entries[0].suite == "runtime-1npu"
    runner = runpy.run_path(str(ROOT / "test/runtime/run_ci_suite.py"))
    assert runner["filter_tests"](entries, DeviceBackend.CUDA, "runtime-1gpu") == (
        [],
        [],
    )
    assert (
        runner["filter_tests"](entries, DeviceBackend.NPU, "runtime-1npu")[0] == entries
    )


def test_runtime_yaml_resolves_through_public_pipeline():
    task = normalize_task(ROOT / "test/ci/ut/ut-runtime-1npu.yaml", ROOT)
    targets = summarize_task_targets(task, ROOT)
    assert "test/runtime/models/test_qwen3_npu_smoke.py" in targets["test_files"]
    matrix = build_matrix(
        ROOT / "test/ci",
        ROOT,
        "per-commit",
        runner_group="npu",
        workflow_stage="unit-test",
    )
    tasks = {item["name"]: item for item in matrix["include"]}
    assert tasks["ut-runtime-1npu"]["npu_runtime"] is True
    assert tasks["npu-operators"]["npu_runtime"] is False
    for group in ("nvidia", "amd"):
        gpu = build_matrix(
            ROOT / "test/ci",
            ROOT,
            "per-commit",
            runner_group=group,
            workflow_stage="unit-test",
        )
        assert all(item["name"] not in tasks for item in gpu["include"])


def test_empty_npu_suite_fails_without_changing_cuda_empty_suite(tmp_path):
    runtime = tmp_path / "test/runtime"
    registry = tmp_path / "test/ci_system"
    runtime.mkdir(parents=True)
    registry.mkdir()
    shutil.copy2(ROOT / "test/runtime/run_ci_suite.py", runtime)
    shutil.copy2(ROOT / "test/ci_system/ci_register.py", registry)
    for device, suite, expected in (
        ("npu", "runtime-1npu", 1),
        ("cuda", "runtime-1gpu", 0),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(runtime / "run_ci_suite.py"),
                "--device",
                device,
                "--suite",
                suite,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize(
    "response, maximum, vocabulary",
    [
        ({"output_ids": [], "meta_info": {}}, 16, 100),
        ({"output_ids": [True], "meta_info": {}}, 16, 100),
        ({"output_ids": [-1], "meta_info": {}}, 16, 100),
        ({"output_ids": [100], "meta_info": {}}, 16, 100),
        ({"output_ids": [1, 2], "meta_info": {}}, 1, 100),
        (
            {
                "output_ids": [1],
                "meta_info": {
                    "completion_tokens": 1,
                    "finish_reason": {"type": "abort"},
                },
            },
            16,
            100,
        ),
        (
            {
                "output_ids": [1],
                "meta_info": {
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                },
            },
            16,
            100,
        ),
    ],
)
def test_invalid_qwen_output_is_a_failure(response, maximum, vocabulary):
    smoke = runpy.run_path(str(SCRIPT))
    with pytest.raises(ValueError):
        smoke["validate_response"](response, maximum, vocabulary)


def test_qwen_accepts_bounded_tokens_and_valid_finish():
    smoke = runpy.run_path(str(SCRIPT))
    response = {
        "output_ids": [1, 2],
        "meta_info": {"completion_tokens": 2, "finish_reason": {"type": "stop"}},
    }
    assert smoke["validate_response"](response, 16, 100) == [1, 2]


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux process ownership and waitid required"
)
@pytest.mark.parametrize(
    "command, timeout, expected",
    [
        ("printf native-ok", "5", 0),
        ("printf native-failure; exit 7", "5", 7),
        ("sleep 30", "1", 124),
    ],
)
def test_native_command_preserves_status_and_releases_lock(
    tmp_path, command, timeout, expected
):
    (tmp_path / "runtime").mkdir()
    lock = tmp_path / "host.lock"
    lock.touch()
    env = dict(
        os.environ,
        NPU_CI_STAGE="ut",
        NPU_CI_WORK=str(tmp_path),
        NPU_CI_TEST_TIMEOUT=timeout,
        NPU_CI_LOCK_TIMEOUT="1",
        NPU_CI_LOCK=str(lock),
        TOKENSPEED_CI_SOURCE_SHA="1" * 40,
    )
    result = run_shell(command, env=env, cwd=tmp_path)
    assert result["returncode"] == expected
    proof = json.loads((tmp_path / "runtime/ut.process.json").read_text())
    assert proof["exit_code"] == expected
    import fcntl

    with lock.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process ownership required")
def test_lock_timeout_does_not_start_a_command(tmp_path):
    import fcntl

    (tmp_path / "runtime").mkdir()
    lock = tmp_path / "host.lock"
    lock.touch()
    env = dict(
        os.environ,
        NPU_CI_STAGE="ut",
        NPU_CI_WORK=str(tmp_path),
        NPU_CI_TEST_TIMEOUT="5",
        NPU_CI_LOCK_TIMEOUT="1",
        NPU_CI_LOCK=str(lock),
        TOKENSPEED_CI_SOURCE_SHA="1" * 40,
    )
    with lock.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = run_shell("touch must-not-start", env=env, cwd=tmp_path)
    assert result["returncode"] != 0
    assert not (tmp_path / "must-not-start").exists()
    assert (
        json.loads((tmp_path / "runtime/ut.process.json").read_text())["child_pid"]
        is None
    )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux parent-death pipe and process groups required",
)
def test_controller_sigkill_reclaims_descendants_and_lock(tmp_path):
    import fcntl
    import shlex
    import signal
    import time

    (tmp_path / "runtime").mkdir()
    lock = tmp_path / "host.lock"
    lock.touch()
    child = tmp_path / "child.py"
    child.write_text(
        "import os, subprocess, sys, time\nfrom pathlib import Path\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nPath('children.json').write_text(__import__('json').dumps([os.getpid(), p.pid]))\ntime.sleep(60)\n"
    )
    controller = tmp_path / "controller.py"
    controller.write_text(
        f"import sys, os\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'test/ci_system')!r})\nfrom npu_runtime import run_shell\nrun_shell({(shlex.quote(sys.executable) + ' ' + shlex.quote(str(child)))!r}, env=dict(os.environ), cwd=Path.cwd())\n"
    )
    env = dict(
        os.environ,
        NPU_CI_STAGE="ut",
        NPU_CI_WORK=str(tmp_path),
        NPU_CI_TEST_TIMEOUT="60",
        NPU_CI_LOCK_TIMEOUT="1",
        NPU_CI_LOCK=str(lock),
        TOKENSPEED_CI_SOURCE_SHA="1" * 40,
    )
    owner = subprocess.Popen(
        [sys.executable, "-B", str(controller)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "children.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "children.json").exists()
        children = json.loads((tmp_path / "children.json").read_text())
        owner.kill()
        owner.wait(timeout=5)
        proof = tmp_path / "runtime/ut.process.json"
        while not proof.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert json.loads(proof.read_text())["exit_code"] != 0
        for pid in children:
            stat = Path(f"/proc/{pid}/stat")
            assert (
                not stat.exists()
                or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
            )
        with lock.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


@pytest.mark.parametrize("status", [0, 7])
def test_runtime_install_entry_preserves_interpreter_arguments_and_status(
    tmp_path, status
):
    task = normalize_task(ROOT / "test/ci/ut/ut-runtime-1npu.yaml", ROOT)
    interpreter = tmp_path / "bootstrap python"
    arguments = tmp_path / "arguments.json"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + f"import json, pathlib, sys\npathlib.Path({str(arguments)!r}).write_text(json.dumps(sys.argv[1:]))\nsys.exit({status})\n"
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", task["install"][0]],
        env=dict(os.environ, NPU_CI_BOOTSTRAP_PYTHON=str(interpreter)),
        cwd=ROOT,
    )
    assert result.returncode == status
    assert json.loads(arguments.read_text()) == [
        "-B",
        "test/ci_system/npu_runtime.py",
        "install",
    ]
