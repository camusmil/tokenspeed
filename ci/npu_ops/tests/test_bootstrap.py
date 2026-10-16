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

"""Execute the workflow bootstrap's deployment guards without NPU hardware."""

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


def bootstrap_script():
    # Execute the actual YAML literal block without adding a YAML dependency.
    workflow = (
        Path(__file__).resolve().parents[3] / ".github/workflows/npu-arm-tests.yml"
    )
    lines = iter(workflow.read_text().splitlines())
    for line in lines:
        if line.strip() == "run: &npu-bootstrap |":
            break
    else:
        raise ValueError("Workflow bootstrap literal block not found")
    block = []
    for line in lines:
        if line.strip() and not line.startswith("          "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block))


class BootstrapTests(unittest.TestCase):
    def run_bootstrap(
        self, repository, allowed_repository, expected_runner, actual_runner, server
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Emulate an unavailable machine type. A valid deployment contract
            # reaches this check and stops before Git, credentials, or hardware.
            uname = root / "uname"
            uname.write_text("#!/bin/sh\necho UnsupportedTestHost\n")
            uname.chmod(0o700)
            summary = root / "summary.md"
            env = dict(
                os.environ,
                PATH=f"{root}{os.pathsep}{os.environ['PATH']}",
                NPU_PHASE="accuracy",
                GITHUB_STEP_SUMMARY=str(summary),
                GITHUB_REPOSITORY=repository,
                GITHUB_SERVER_URL=server,
                TOKENSPEED_NPU_CI_REPOSITORY=allowed_repository,
                TOKENSPEED_NPU_ARM_RUNNER_NAME=expected_runner,
                RUNNER_NAME=actual_runner,
            )
            result = subprocess.run(
                ["bash", "-c", bootstrap_script()],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            return result.stdout, summary.read_text()

    def test_official_repository_needs_no_fork_allowlist(self):
        output, summary = self.run_bootstrap(
            "lightseekorg/tokenspeed",
            "",
            "npu-arm-test",
            "npu-arm-test",
            "https://github.com",
        )
        self.assertIn("system 不符合部署契约", output)
        self.assertIn("`platform`", summary)

    def test_explicit_fork_allowlist_passes_deployment_contract(self):
        output, summary = self.run_bootstrap(
            "example/tokenspeed",
            "example/tokenspeed",
            "npu-arm-test",
            "npu-arm-test",
            "https://github.com",
        )
        self.assertIn("system 不符合部署契约", output)
        self.assertIn("`platform`", summary)

    def test_missing_runner_name_fails_before_platform(self):
        output, summary = self.run_bootstrap(
            "lightseekorg/tokenspeed", "", "", "npu-arm-test", "https://github.com"
        )
        self.assertIn("必须配置仓库变量 TOKENSPEED_NPU_ARM_RUNNER_NAME", output)
        self.assertIn("`deployment-contract`", summary)

    def test_wrong_runner_name_fails(self):
        output, summary = self.run_bootstrap(
            "lightseekorg/tokenspeed",
            "",
            "expected-npu",
            "wrong-npu",
            "https://github.com",
        )
        self.assertIn("runner 不符合部署契约", output)
        self.assertIn("`deployment-contract`", summary)

    def test_unlisted_repository_fails(self):
        output, summary = self.run_bootstrap(
            "example/tokenspeed",
            "",
            "npu-arm-test",
            "npu-arm-test",
            "https://github.com",
        )
        self.assertIn("GITHUB_REPOSITORY 未在仓库白名单中", output)
        self.assertIn("`deployment-contract`", summary)

    def test_even_allowlisted_invalid_repository_fails(self):
        for repository in ("example/../other", "example/..", "example/repo?query=1"):
            with self.subTest(repository=repository):
                output, summary = self.run_bootstrap(
                    repository,
                    repository,
                    "npu-arm-test",
                    "npu-arm-test",
                    "https://github.com",
                )
                self.assertIn("GITHUB_REPOSITORY", output)
                self.assertIn("无效", output)
                self.assertIn("`deployment-contract`", summary)

    def test_non_github_server_fails(self):
        output, summary = self.run_bootstrap(
            "lightseekorg/tokenspeed",
            "",
            "npu-arm-test",
            "npu-arm-test",
            "https://example.invalid",
        )
        self.assertIn("server 不符合部署契约", output)
        self.assertIn("`deployment-contract`", summary)


if __name__ == "__main__":
    unittest.main()
