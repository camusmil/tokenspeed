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

"""NPU GitHub reads and preparation of the workflow's source reporter."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def run_github(command, *, cwd, timeout_seconds, capture_output):
    for attempt in range(1, 5):
        try:
            return subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout_seconds,
                capture_output=capture_output,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, subprocess.CalledProcessError) and exc.returncode in (
                -2,
                -15,
                130,
                143,
            ):
                raise
            error = exc.stderr
            if error:
                print(
                    (
                        error.decode(errors="replace")
                        if isinstance(error, bytes)
                        else error
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            print(
                f"GitHub read attempt {attempt}/4 failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == 4:
                raise
            print("Retrying GitHub read in 30 seconds", file=sys.stderr, flush=True)
            time.sleep(30)
    raise AssertionError("Unreachable retry state")


def prepare_reporter(repo: Path, workflow_sha: str, directory: Path, github_env: Path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-env", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_reporter(args.repo, args.workflow_sha, args.directory, args.github_env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
        ) from None
