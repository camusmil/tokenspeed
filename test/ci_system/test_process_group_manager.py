import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import process_group_manager as pgm
import pytest
from process_group_manager import CommandSupervision, ProcessGroupManager


def test_cleanup_stale_kills_all_persisted_process_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path)
    monkeypatch.setattr(pgm.time, "sleep", lambda _seconds: None)

    killed = []
    monkeypatch.setattr(
        pgm.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )

    pgm._add_pgid("gb200-runner", 111)
    pgm._add_pgid("gb200-runner", 222)

    manager = ProcessGroupManager("gb200-runner", supervision=None)
    manager.cleanup_stale()

    assert killed == [
        (111, signal.SIGTERM),
        (222, signal.SIGTERM),
        (111, signal.SIGKILL),
        (222, signal.SIGKILL),
    ]
    assert not pgm._pgid_path("gb200-runner").exists()


def test_terminate_all_kills_persisted_process_groups_without_tracked_parents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path)
    monkeypatch.setattr(pgm.time, "sleep", lambda _seconds: None)

    killed = []
    monkeypatch.setattr(
        pgm.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )

    pgm._add_pgid("gb200-runner", 111)
    pgm._add_pgid("gb200-runner", 222)

    manager = ProcessGroupManager("gb200-runner", supervision=None)
    manager.terminate_all()

    assert killed == [
        (111, signal.SIGTERM),
        (222, signal.SIGTERM),
        (111, signal.SIGKILL),
        (222, signal.SIGKILL),
    ]
    assert not pgm._pgid_path("gb200-runner").exists()


ROOT = Path(__file__).resolve().parents[2]


def supervised_manager(env):
    return ProcessGroupManager(
        "test-owner",
        supervision=CommandSupervision(
            log_directory=Path(env["NPU_CI_WORK"]) / "runtime",
            stage_key="NPU_CI_STAGE",
            timeouts={"ut": int(env["NPU_CI_TEST_TIMEOUT"])},
            lock_path=Path(env["NPU_CI_LOCK"]),
            lock_wait=int(env["NPU_CI_LOCK_TIMEOUT"]),
            source_sha=env["TOKENSPEED_CI_SOURCE_SHA"],
            owner_pid=env.get("NPU_CI_WORKER_PID"),
            owner_start=env.get("NPU_CI_WORKER_START"),
        ),
    )


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
    result = supervised_manager(env).run(command, env=env, cwd=tmp_path, check=False)
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
        result = supervised_manager(env).run(
            "touch must-not-start", env=env, cwd=tmp_path, check=False
        )
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
        f"import sys, os\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'test/ci_system')!r})\nfrom test_process_group_manager import supervised_manager\nsupervised_manager(dict(os.environ)).run({(shlex.quote(sys.executable) + ' ' + shlex.quote(str(child)))!r}, env=dict(os.environ), cwd=Path.cwd())\n"
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


@pytest.fixture
def command_env(tmp_path):
    (tmp_path / "runtime").mkdir()
    lock = tmp_path / "host.lock"
    lock.touch()
    return dict(
        os.environ,
        NPU_CI_STAGE="ut",
        NPU_CI_WORK=str(tmp_path),
        NPU_CI_TEST_TIMEOUT="5",
        NPU_CI_LOCK_TIMEOUT="5",
        NPU_CI_LOCK=str(lock),
        TOKENSPEED_CI_SOURCE_SHA="1" * 40,
    )


def assert_lock_released(env):
    import fcntl

    with open(env["NPU_CI_LOCK"], "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "command, expected",
    [
        ("printf 'Test Summary: 1/1 passed\\n'; kill -TERM $$", 0),
        ("printf 'Test Summary: 1/2 passed\\n'; kill -TERM $$", -15),
        ("exit 7", 7),
    ],
)
def test_gpu_command_exit_policy_is_unchanged(tmp_path, monkeypatch, command, expected):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path / "ledger")
    manager = ProcessGroupManager("gpu-regression", supervision=None)
    result = manager.run(command, cwd=tmp_path, env=dict(os.environ), check=False)
    assert result["returncode"] == expected
    manager.terminate_all()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor required")
def test_supervised_success_text_cannot_hide_signal_failure(tmp_path, command_env):
    manager = supervised_manager(command_env)
    with pytest.raises(RuntimeError, match="exit code 143"):
        manager.run(
            "printf 'Test Summary: 1/1 passed\\n'; kill -TERM $$",
            cwd=tmp_path,
            env=command_env,
        )
    assert_lock_released(command_env)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor required")
@pytest.mark.parametrize("failure", ["existing-log", "unwritable-proof"])
def test_supervised_evidence_failure_still_releases_lock(
    tmp_path, command_env, failure
):
    if failure == "existing-log":
        (tmp_path / "runtime/ut.log").touch()
        with pytest.raises(RuntimeError, match="exit code 1"):
            supervised_manager(command_env).run(
                "sleep 30", cwd=tmp_path, env=command_env
            )
    else:
        (tmp_path / "runtime/ut.process.json").mkdir()
        with pytest.raises(IsADirectoryError):
            supervised_manager(command_env).run("exit 0", cwd=tmp_path, env=command_env)
    assert_lock_released(command_env)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owner identity required")
def test_reused_owner_pid_does_not_start_command(tmp_path, command_env):
    command_env.update(
        NPU_CI_WORKER_PID=str(os.getpid()), NPU_CI_WORKER_START="invalid"
    )
    result = supervised_manager(command_env).run(
        "touch must-not-start", cwd=tmp_path, env=command_env, check=False
    )
    assert result["returncode"] != 0
    assert not (tmp_path / "must-not-start").exists()
    assert_lock_released(command_env)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux flock required")
def test_execution_deadline_starts_after_lock_acquisition(tmp_path, command_env):
    import fcntl
    import threading
    import time

    command_env["NPU_CI_TEST_TIMEOUT"] = "1"
    handle = open(command_env["NPU_CI_LOCK"], "r+")
    fcntl.flock(handle, fcntl.LOCK_EX)

    def release():
        time.sleep(1.5)
        fcntl.flock(handle, fcntl.LOCK_UN)

    thread = threading.Thread(target=release)
    thread.start()
    try:
        result = supervised_manager(command_env).run(
            "sleep 0.2; printf acquired", cwd=tmp_path, env=command_env
        )
        assert result["returncode"] == 0
        assert "acquired" in result["output"]
    finally:
        thread.join()
        handle.close()
    assert_lock_released(command_env)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descendant cleanup required")
@pytest.mark.parametrize(
    "command, expected",
    [
        ("sleep 60 & echo $! > descendant; exit 0", 0),
        ("sleep 60 & echo $! > descendant; wait", 124),
    ],
)
def test_cleanup_targets_only_owned_descendants(
    tmp_path, command_env, command, expected
):
    command_env["NPU_CI_TEST_TIMEOUT"] = "1"
    unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        result = supervised_manager(command_env).run(
            command, cwd=tmp_path, env=command_env, check=False
        )
        assert result["returncode"] == expected
        pid = int((tmp_path / "descendant").read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert (
            not stat.exists() or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
        )
        assert unrelated.poll() is None
        assert_lock_released(command_env)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_supervised_cleanup_never_reads_persistent_runner_ledger(
    tmp_path, command_env, monkeypatch
):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path / "ledger")
    pgm._add_pgid("test-owner", 123456)
    manager = supervised_manager(command_env)
    manager.terminate_all()
    assert pgm._load_pgids("test-owner") == {123456}
    with pytest.raises(ValueError, match="other runner jobs"):
        manager.cleanup_stale()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker lifetime required")
def test_worker_exit_cancels_command_and_releases_lock(tmp_path, command_env):
    import threading
    import time

    worker = subprocess.Popen(["sleep", "60"])
    command_env.update(
        NPU_CI_WORKER_PID=str(worker.pid),
        NPU_CI_WORKER_START=pgm._process_start(worker.pid),
    )

    def stop_worker():
        deadline = time.monotonic() + 5
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        worker.terminate()
        worker.wait(timeout=5)

    thread = threading.Thread(target=stop_worker)
    thread.start()
    try:
        result = supervised_manager(command_env).run(
            "touch started; sleep 60", cwd=tmp_path, env=command_env, check=False
        )
        assert result["returncode"] == 143
        assert_lock_released(command_env)
    finally:
        thread.join(timeout=10)
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
