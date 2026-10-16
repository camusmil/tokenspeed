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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from changes import load_manifest, select_suite


class ChangeSelectionTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.temp = None
        self.repo = None
        self.source = Path(__file__).resolve().parents[3]
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
            self.git("rev-parse", "HEAD"),
            "refs/heads/test",
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
            "tokenspeed-kernel/python/a.py",
            "python/pyproject.toml",
        ]
        for path in paths:
            with self.subTest(path=path):
                before = self.git("rev-parse", "HEAD")
                self.write(path, "test\n")
                result = self.scan(before, self.commit())
                self.assert_suite_selected(result)
                self.assertEqual(result["matched_paths_sample"], [path])

    def test_modify_then_delete_operator_file(self):
        path = "tokenspeed-kernel-npu/a.py"
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
        path = "tokenspeed-kernel-npu/a space\n\u4e2d\u6587.py"
        self.write(path, "test\n")
        result = self.scan(self.base, self.commit())
        self.assert_suite_selected(result)
        self.assertEqual(result["matched_paths_sample"], [path])

    def test_ci_infrastructure_changes(self):
        for path in (
            "ci/npu_ops/scatter/accuracy.py",
            ".github/workflows/pr-test-ascendnpu-arm.yml",
            "ci/npu_ops/tests/test_changes.py",
            "test/ci_system/install_triton_ascend.sh",
        ):
            with self.subTest(path=path):
                before = self.git("rev-parse", "HEAD")
                self.write(path, "test\n")
                result = self.scan(before, self.commit())
                self.assert_suite_selected(result)
                self.assertEqual(result["reason"], "watched-path-changed")
                self.assertEqual(result["matched_paths_sample"], [path])

    def test_missing_git_object_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.scan("f" * 40, self.base)

    def test_new_branch_selects_suite(self):
        result = self.scan("0" * 40, self.base)
        self.assert_suite_selected(result)
        self.assertEqual(result["reason"], "new-branch")

    def test_invalid_sha_or_ref_fails(self):
        for before in ("invalid", None, "--help", "a" * 64):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.scan(before, self.base)
        for ref in ("--help", "refs/heads/bad..name"):
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                select_suite(
                    self.repo, self.manifest, "workflow_dispatch", {}, self.base, ref
                )

    def test_manual_selects_suite_without_git_diff(self):
        result = select_suite(
            self.repo,
            self.manifest,
            "workflow_dispatch",
            {},
            self.base,
            "refs/heads/test",
        )
        self.assert_suite_selected(result)
        self.assertEqual(result["reason"], "manual")

    def test_tag_and_branch_deletion_skip(self):
        for ref, deleted in (("refs/tags/v1", False), ("refs/heads/test", True)):
            event = {
                "ref": ref,
                "deleted": deleted,
                "before": self.base,
                "after": self.base,
            }
            result = select_suite(
                self.repo, self.manifest, "push", event, self.base, ref
            )
            self.assertFalse(result["should_run"])

    def test_missing_ref_is_an_error(self):
        with self.assertRaises(ValueError):
            select_suite(
                self.repo, self.manifest, "push", {}, self.base, "refs/heads/test"
            )

    def test_docs_and_other_backends_skip(self):
        for path in (
            "ci/npu_ops/README.md",
            "tokenspeed-kernel-npu/README.md",
            "tokenspeed-kernel-amd/python/a.py",
            "docs/npu-operator-ci.md",
        ):
            self.write(path, "documentation or other vendor")
        self.assertFalse(self.scan(self.base, self.commit())["should_run"])

    def pr_fixture(self, path):
        self.git("checkout", "-qb", "feature", self.base)
        self.write(path, "PR change")
        head = self.commit()
        self.git("checkout", "-qb", "target", self.base)
        # A watched file introduced only on base must not select this PR.
        self.write("tokenspeed-kernel-npu/base-only.py", "base change")
        base = self.commit()
        self.git("merge", "--no-ff", "-qm", "merge fixture", head)
        merge = self.git("rev-parse", "HEAD")
        return {
            "number": 7,
            "pull_request": {
                "base": {"sha": base, "ref": "target"},
                "head": {"sha": head, "ref": "feature"},
            },
        }, merge

    def test_pr_merge_base_diff_ignores_base_only_changes(self):
        event, merge = self.pr_fixture("docs/pr.md")
        result = select_suite(
            self.repo, self.manifest, "pull_request", event, merge, "refs/pull/7/merge"
        )
        self.assertFalse(result["should_run"])
        self.assertEqual(result["changed_files"], 1)

    def test_pr_operator_change_selects_and_wrong_merge_fails(self):
        event, merge = self.pr_fixture("tokenspeed-kernel-npu/pr.py")
        result = select_suite(
            self.repo, self.manifest, "pull_request", event, merge, "refs/pull/7/merge"
        )
        self.assert_suite_selected(result)
        event["pull_request"]["head"]["sha"] = self.base
        with self.assertRaises(ValueError):
            select_suite(
                self.repo,
                self.manifest,
                "pull_request",
                event,
                merge,
                "refs/pull/7/merge",
            )

    def test_cli_writes_action_outputs(self):
        event = self.repo / "event.json"
        event.write_text("{}", encoding="utf-8")
        output = self.repo / "output.txt"
        script = Path(__file__).resolve().parents[1] / "changes.py"
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
                "--checkout-sha",
                self.base,
                "--checkout-ref",
                "refs/heads/test",
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
