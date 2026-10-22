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

"""Exercise acquisition, retries and offline provenance with real Git repositories."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ci.npu_ops_test import github
from ci.npu_ops_test.changes import github_json

ROOT = Path(__file__).resolve().parents[2]


def step_script(workflow, job, name):
    data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return next(s["run"] for s in data["jobs"][job]["steps"] if s.get("name") == name)


@pytest.fixture
def source(tmp_path):
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
    for name in ("ci/npu_ops_test/github.py", ".github/scripts/ci_result_source.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    git("add", ".")
    git("commit", "--quiet", "-m", "workflow reporter")
    workflow = git("rev-parse", "HEAD")
    (repo / ".github/scripts/ci_result_source.py").write_text(
        "raise RuntimeError('candidate reporter must not run')\n"
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
    source, hardware, failures, status, tmp_path
):
    env, calls, waits, workflow, candidate = source
    env.update(FETCH_FAILURES=str(failures), FETCH_STATUS=str(status))
    if hardware:
        env["NPU_REPORTER_DIR"] = str(tmp_path / "task")
    script = step_script(
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


def test_missing_workflow_commit_is_fetched_before_tests(source, tmp_path):
    env, calls, waits, workflow, candidate = source
    env.update(NPU_REPORTER_DIR=str(tmp_path / "task"), NPU_WORKFLOW_SHA=workflow)
    script = step_script(
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


def test_reporter_is_offline_pinned_and_rejects_tampering(source, tmp_path):
    env, calls, waits, workflow, candidate = source
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
    github.prepare_reporter(repo, workflow, tmp_path / "task", Path(env["GITHUB_ENV"]))
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
    record = step_script(
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


@pytest.mark.parametrize("failures,status", [(3, 1), (4, 1), (1, 130), (1, 143)])
def test_api_retry_and_original_failure(monkeypatch, tmp_path, failures, status):
    calls = tmp_path / "api-calls"
    gh = tmp_path / "gh"
    gh.write_text(f"#!{sys.executable}\n" + f"""
import pathlib, sys
path = pathlib.Path({str(calls)!r})
count = int(path.read_text()) + 1 if path.exists() else 1
path.write_text(str(count))
if count <= {failures}:
    print('original API transport failure', file=sys.stderr)
    sys.exit({status})
print('{{"ok": true}}')
""")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    waits = []
    monkeypatch.setattr(github.time, "sleep", waits.append)
    if failures == 4 or status in (130, 143):
        with pytest.raises(subprocess.CalledProcessError) as error:
            github_json("/repos/fixture/tokenspeed/pulls/12", paginate=False)
        assert error.value.returncode == status
        assert "original API transport failure" in error.value.stderr
    else:
        assert github_json("/repos/fixture/tokenspeed/pulls/12", paginate=False) == {
            "ok": True
        }
    assert int(calls.read_text()) == (1 if status in (130, 143) else 4)
    assert waits == ([] if status in (130, 143) else [30, 30, 30])


def test_request_timeout_is_retried_and_remains_failure(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(github.time, "sleep", waits.append)
    with pytest.raises(subprocess.TimeoutExpired):
        github.run_github(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout_seconds=0.05,
            capture_output=True,
        )
    assert waits == [30, 30, 30]
