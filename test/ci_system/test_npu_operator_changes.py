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

"""Exercise change selection with real commits in isolated Git repositories."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from npu_operator_changes import load_manifest, select_suite


class ChangeSelectionTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.temp = None
        self.repo = None
        self.source = Path(__file__).resolve().parents[2]
        self.manifest_path = self.source / "ci/npu_ops/manifest.json"
        self.manifest = {}
        self.base = ""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.manifest = load_manifest(self.manifest_path)
        self.git("init", "-q")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci-test@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("config", "commit.gpgsign", "false")
        self.write("docs/start.md", "initial\n")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, stderr=subprocess.PIPE, text=True
        ).strip()

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def scan(self, before, after):
        return select_suite(
            self.repo,
            self.manifest,
            "push",
            {"ref": "refs/heads/test", "before": before, "after": after},
        )

    def assert_suite_selected(self, result):
        self.assertTrue(result["should_run"])
        self.assertNotIn("matrix", result)

    def test_unrelated_and_empty_changes_skip(self):
        self.write("docs/start.md", "updated\n")
        head = self.commit()
        for before in (self.base, head):
            result = self.scan(before, head)
            self.assertFalse(result["should_run"])
            self.assertNotIn("matrix", result)

    def test_additions_across_operator_roots(self):
        paths = [
            "tokenspeed-kernel-npu/python/a.py",
            "tokenspeed-kernel-amd/README.md",
            "tokenspeed-kernel/python/a.py",
            "tokenspeed-mla/test/a.py",
            "python/tokenspeed/runtime/execution/a.py",
            "new_package/ops/a.py",
            "ops/a.py",
            "kernels/a.py",
            "new_package/kernels/a.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                before = self.git("rev-parse", "HEAD")
                self.write(path, "test\n")
                result = self.scan(before, self.commit())
                self.assert_suite_selected(result)
                self.assertEqual(result["matched_paths_sample"], [path])

    def test_modify_then_delete_operator_file(self):
        path = "tokenspeed-kernel-npu/README.md"
        self.write(path, "before\n")
        before = self.commit()
        self.write(path, "after\n")
        after = self.commit()
        self.assert_suite_selected(self.scan(before, after))
        (self.repo / path).unlink()
        self.assert_suite_selected(self.scan(after, self.commit()))

    def test_rename_out_of_and_into_watched_directory(self):
        watched = "tokenspeed-kernel-npu/a.py"
        outside = "docs/moved.py"
        self.write(watched, "same content\n")
        before = self.commit()
        self.git("mv", watched, outside)
        after = self.commit()
        result = self.scan(before, after)
        self.assert_suite_selected(result)
        self.assertIn(watched, result["matched_paths_sample"])
        self.git("mv", outside, watched)
        self.assert_suite_selected(self.scan(after, self.commit()))

    def test_more_than_three_thousand_changed_paths(self):
        for i in range(3100):
            self.write(f"docs/bulk/{i:04}.txt", "data\n")
        self.write("tokenspeed-kernel-npu/z.py", "test\n")
        result = self.scan(self.base, self.commit())
        self.assert_suite_selected(result)
        self.assertEqual(result["changed_files"], 3101)

    def test_paths_with_whitespace_and_unicode(self):
        path = "tokenspeed-kernel-npu/a space\n中文.py"
        self.write(path, "test\n")
        result = self.scan(self.base, self.commit())
        self.assert_suite_selected(result)
        self.assertEqual(result["matched_paths_sample"], [path])

    def test_ci_infrastructure_changes(self):
        for path in (
            "ci/npu_ops/scatter/accuracy.py",
            ".github/workflows/npu-tests.yml",
            "test/ci_system/test_npu_operator_changes.py",
            "test/ci_system/install_triton_ascend.sh",
        ):
            with self.subTest(path=path):
                before = self.git("rev-parse", "HEAD")
                self.write(path, "test\n")
                self.assert_suite_selected(self.scan(before, self.commit()))

    def test_missing_git_object_selects_suite(self):
        result = self.scan("f" * 40, self.base)
        self.assert_suite_selected(result)
        self.assertEqual(result["reason"], "git-diff-unavailable")

    def test_new_branch_or_invalid_comparison_selects_suite(self):
        for before in ("0" * 40, "invalid", None):
            result = self.scan(before, self.base)
            self.assert_suite_selected(result)
            self.assertEqual(result["reason"], "comparison-unavailable")

    def test_manual_selects_suite_without_git_diff(self):
        result = select_suite(self.repo, self.manifest, "workflow_dispatch", {})
        self.assert_suite_selected(result)
        self.assertEqual(result["reason"], "manual")

    def test_tag_and_branch_deletion_skip(self):
        for event in (
            {"ref": "refs/tags/v1"},
            {"ref": "refs/heads/test", "deleted": True},
        ):
            result = select_suite(self.repo, self.manifest, "push", event)
            self.assertFalse(result["should_run"])
            self.assertNotIn("matrix", result)

    def test_missing_ref_is_an_error(self):
        with self.assertRaises(ValueError):
            select_suite(self.repo, self.manifest, "push", {})

    def test_cli_writes_action_outputs(self):
        event = self.repo / "event.json"
        event.write_text("{}", encoding="utf-8")
        output = self.repo / "output.txt"
        script = Path(__file__).with_name("npu_operator_changes.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--repo-root",
                str(self.repo),
                "--manifest",
                str(self.manifest_path),
                "--event-name",
                "workflow_dispatch",
                "--event-path",
                str(event),
                "--github-output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assert_suite_selected(json.loads(completed.stdout))
        lines = dict(line.split("=", 1) for line in output.read_text().splitlines())
        self.assertEqual(lines["should_run"], "true")
        self.assertEqual(lines, {"should_run": "true", "reason": "manual"})

    def test_invalid_manifest_fails(self):
        invalid = self.repo / "invalid.json"
        for changes in (
            {"schema_version": 1},
            {"watch_paths": []},
            {"operators": []},
            {"operators": self.manifest["operators"] * 2},
        ):
            invalid.write_text(
                json.dumps({**self.manifest, **changes}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
