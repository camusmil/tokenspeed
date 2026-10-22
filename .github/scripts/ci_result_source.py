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

"""Record the source actually checked out by a native CI job."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def prepare_reporter(repo: Path, workflow_sha: str, directory: Path, github_env: Path):
    # Only preparation needs repository helpers; the copied reporter stays standalone.
    sys.path.insert(0, str(repo.resolve()))
    from ci.npu_ops_test.changes import run_github

    if not re.fullmatch(r"[0-9a-f]{40}", workflow_sha):
        raise ValueError("The workflow reporter requires an immutable commit SHA")
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{workflow_sha}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if present.returncode:
        run_github(
            [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "http.extraheader=",
                "fetch",
                "--no-tags",
                "--depth=1",
                "origin",
                workflow_sha,
            ],
            cwd=repo,
            timeout_seconds=None,
            capture_output=False,
        )
    source = f"{workflow_sha}:.github/scripts/ci_result_source.py"
    blob = subprocess.check_output(
        ["git", "rev-parse", source], cwd=repo, text=True
    ).strip()
    data = subprocess.check_output(["git", "show", source], cwd=repo)
    directory.mkdir(parents=True, exist_ok=True)
    reporter = directory / "source-reporter.py"
    reporter.write_bytes(data)
    reporter.chmod(0o400)
    with github_env.open("a", encoding="utf-8") as output:
        output.write(
            f"NPU_SOURCE_REPORTER={reporter}\nNPU_SOURCE_REPORTER_BLOB={blob}\n"
        )
    print(
        f"Prepared source reporter from workflow commit {workflow_sha}, blob {blob}",
        flush=True,
    )


def write_source(work: Path, expected: str, config: str, runner: str) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", actual) or (
        re.fullmatch(r"[0-9a-f]{40}", expected) and expected != actual
    ):
        raise ValueError("CI source differs from selected commit")
    target = work / ".ci-artifacts/source.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps({"source_sha": actual, "config": config, "runner": runner})
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare"]:
        parser = argparse.ArgumentParser(
            description="Prepare an offline source reporter"
        )
        parser.add_argument("--repo", type=Path, required=True)
        parser.add_argument("--workflow-sha", required=True)
        parser.add_argument("--directory", type=Path, required=True)
        parser.add_argument("--github-env", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        try:
            prepare_reporter(
                args.repo, args.workflow_sha, args.directory, args.github_env
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
            ) from None
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    args = parser.parse_args()
    try:
        write_source(args.work_dir, args.expected, args.config, args.runner)
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise SystemExit("CI source verification failed.") from None
