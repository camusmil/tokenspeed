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

"""Exercise the workflow fetch retry helper with local Git and real waits."""

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/pr-test-ascendnpu-arm.yml"


class FetchRetryTest(unittest.TestCase):
    def test_recovery_exhaustion_and_cancellation(self):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        bootstrap = workflow["jobs"]["npu"]["steps"][0]["run"]
        start = bootstrap.index("fetch_source() {")
        end = bootstrap.index("\n}", start) + 2
        helper = bootstrap[start:end]
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            env = os.environ.copy()
            env.update(
                GIT_AUTHOR_NAME="CI Retry Test",
                GIT_AUTHOR_EMAIL="ci-retry@example.com",
                GIT_COMMITTER_NAME="CI Retry Test",
                GIT_COMMITTER_EMAIL="ci-retry@example.com",
            )
            for args in (
                ["init", "-q", str(source)],
                [
                    "-C",
                    str(source),
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "Retry transport fixture",
                ],
            ):
                subprocess.run(
                    [real_git, *args],
                    env=env,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            sha = subprocess.check_output(
                [real_git, "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            scenarios = (
                ("first_success", 0, 128, 1, 0),
                ("second_success", 1, 128, 2, 0),
                ("fourth_success", 3, 128, 4, 0),
                ("exhausted", 4, 128, 4, 128),
                ("interrupted", 4, 130, 1, 130),
                ("terminated", 4, 143, 1, 143),
            )

            def run_case(scenario):
                name, failures, failure_code, attempts, expected_code = scenario
                work = root / name
                checkout = work / "checkout"
                checkout.mkdir(parents=True)
                subprocess.run(
                    [real_git, "init", "-q", str(checkout)],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                bindir = work / "bin"
                bindir.mkdir()
                shim = bindir / "git"
                # Only the remote transport is replaced; successful fetches use
                # a real local Git server, and retry waits are the real sleep.
                shim.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                    import json, os, sys, time
                    from pathlib import Path
                    args = sys.argv[1:]
                    events = Path(os.environ['FETCH_EVENTS'])
                    calls = len(events.read_text().splitlines()) if events.exists() else 0
                    with events.open('a') as output:
                        print(json.dumps({'at': time.monotonic()}), file=output)
                    if calls < int(os.environ['FETCH_FAILURES']):
                        print('fatal: simulated transport failure', file=sys.stderr)
                        sys.exit(int(os.environ['FETCH_FAILURE_CODE']))
                    args[-2] = os.environ['FETCH_LOCAL_REMOTE']
                    os.execv(os.environ['FETCH_REAL_GIT'], [os.environ['FETCH_REAL_GIT'], *args])
                    """))
                shim.chmod(0o755)
                case_env = env.copy()
                case_env.update(
                    PATH=f"{bindir}{os.pathsep}{env['PATH']}",
                    FETCH_EVENTS=str(work / "events.jsonl"),
                    FETCH_FAILURES=str(failures),
                    FETCH_FAILURE_CODE=str(failure_code),
                    FETCH_LOCAL_REMOTE=source.as_uri(),
                    FETCH_REAL_GIT=real_git,
                    GITHUB_SHA=sha,
                    GITHUB_REPOSITORY="camusmil/tokenspeed",
                    work=str(work),
                )
                script = "set -Eeuo pipefail\n" + helper + "\nfetch_source\n"
                script += 'touch "$work/after-fetch"\n'
                started = time.monotonic()
                result = subprocess.run(
                    ["bash", "-c", script],
                    env=case_env,
                    text=True,
                    capture_output=True,
                    timeout=150,
                )
                elapsed = time.monotonic() - started
                self.assertEqual(result.returncode, expected_code, result.stderr)
                events = [
                    json.loads(line)
                    for line in (work / "events.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(events), attempts, result.stdout)
                for previous, current in zip(events, events[1:]):
                    self.assertGreaterEqual(current["at"] - previous["at"], 29.9)
                self.assertEqual((work / "after-fetch").exists(), expected_code == 0)
                if expected_code == 0:
                    actual = subprocess.check_output(
                        [real_git, "-C", str(checkout), "rev-parse", "FETCH_HEAD"],
                        text=True,
                    ).strip()
                    self.assertEqual(actual, sha)
                else:
                    self.assertIn("simulated transport failure", result.stderr)
                return {
                    "scenario": name,
                    "attempts": len(events),
                    "exit": result.returncode,
                    "elapsed_seconds": round(elapsed, 3),
                }

            # Independent cases run together so real 30-second waits cost
            # about 90 seconds overall rather than several minutes.
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(run_case, scenarios))
            print(json.dumps(results), flush=True)


if __name__ == "__main__":
    unittest.main()
