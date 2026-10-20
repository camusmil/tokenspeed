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

"""CPU-only tests for NPU selection from GitHub API changed paths."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci.npu_ops_test.changes import load_manifest, select_suite

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ci/npu_ops_test/manifest.json"
SELECTOR = ROOT / "ci/npu_ops_test/changes.py"


class ChangeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST)

    def test_watched_paths_select_all_operators(self):
        for event in ("pull_request", "push"):
            for path in (
                "tokenspeed-kernel-npu/python/ops.py",
                "tokenspeed-kernel/python/shared.py",
                "ci/npu_ops_test/changes.py",
                ".github/workflows/pr-test-ascendnpu-arm.yml",
            ):
                with self.subTest(event=event, path=path):
                    result = select_suite(self.manifest, event, {path})
                    self.assertTrue(result["should_run"])
                    self.assertEqual(result["reason"], "watched-path-changed")
                    self.assertEqual(result["matched_paths_sample"], [path])

    def test_unrelated_and_documentation_changes_skip(self):
        for event in ("pull_request", "push"):
            for paths in (
                set(),
                {"python/tokenspeed/runtime.py"},
                {"tokenspeed-kernel-amd/source.py"},
                {"tokenspeed-kernel-npu/README.md", "ci/npu_ops_test/guide.RST"},
            ):
                with self.subTest(event=event, paths=paths):
                    result = select_suite(self.manifest, event, paths)
                    self.assertFalse(result["should_run"])
                    self.assertEqual(result["reason"], "no-watched-changes")
                    self.assertEqual(result["matched_files"], 0)

    def test_manual_always_runs(self):
        result = select_suite(self.manifest, "workflow_dispatch", set())
        self.assertTrue(result["should_run"])
        self.assertEqual(result["reason"], "manual")

    def test_rename_into_or_out_of_watched_path_runs(self):
        # The API supplies both filename and previous_filename.
        for paths in (
            {"misc/new.py", "tokenspeed-kernel-npu/old.py"},
            {"tokenspeed-kernel-npu/new.py", "misc/old.py"},
        ):
            with self.subTest(paths=paths):
                result = select_suite(self.manifest, "pull_request", paths)
                self.assertTrue(result["should_run"])
                self.assertEqual(result["changed_files"], 2)
                self.assertEqual(result["matched_files"], 1)

    def test_deleted_path_needs_no_local_file(self):
        result = select_suite(
            self.manifest, "push", {"tokenspeed-kernel-npu/deleted.py"}
        )
        self.assertTrue(result["should_run"])

    def test_sample_is_bounded_without_truncating_classification(self):
        paths = {f"misc/{index}.py" for index in range(3000)}
        paths.update(f"tokenspeed-kernel-npu/{index}.py" for index in range(30))
        result = select_suite(self.manifest, "pull_request", paths)
        self.assertTrue(result["should_run"])
        self.assertEqual(result["changed_files"], 3030)
        self.assertEqual(result["matched_files"], 30)
        self.assertEqual(len(result["matched_paths_sample"]), 20)

    def test_matching_is_case_sensitive(self):
        result = select_suite(
            self.manifest, "pull_request", {"TokenSpeed-kernel-npu/source.py"}
        )
        self.assertFalse(result["should_run"])

    def test_unsupported_event_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported event"):
            select_suite(self.manifest, "pull_request_target", set())

    def run_cli(self, paths_file, manifest, output):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SELECTOR),
                "--repo-root",
                str(ROOT),
                "--manifest",
                str(manifest),
                "--event-name",
                "pull_request",
                "--changed-files",
                str(paths_file),
                "--github-output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_cli_deduplicates_api_paths_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            paths = temp / "paths.txt"
            paths.write_text(
                "\ntokenspeed-kernel-npu/removed.py\n"
                "tokenspeed-kernel-npu/removed.py\nmisc/new.py\n"
            )
            output = temp / "outputs.txt"
            process = self.run_cli(paths, MANIFEST, output)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["should_run"])
            self.assertEqual(result["changed_files"], 2)
            self.assertEqual(
                output.read_text(), "should_run=true\nreason=watched-path-changed\n"
            )

    def test_cli_missing_api_path_file_fails_without_skip_output(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            output = temp / "outputs.txt"
            process = self.run_cli(temp / "missing.txt", MANIFEST, output)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("NPU change scan failed", process.stderr)
            self.assertFalse(output.exists())

    def test_cli_invalid_manifest_fails_without_skip_output(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            paths = temp / "paths.txt"
            paths.write_text("docs/readme.md\n")
            manifest = temp / "manifest.json"
            manifest.write_text('{"schema_version": 2}')
            output = temp / "outputs.txt"
            process = self.run_cli(paths, manifest, output)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("schema_version=3", process.stderr)
            self.assertFalse(output.exists())

    def test_cli_required_output_write_failure_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            paths = temp / "paths.txt"
            paths.write_text("tokenspeed-kernel-npu/source.py\n")
            process = self.run_cli(paths, MANIFEST, temp / "missing" / "out.txt")
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("NPU change scan failed", process.stderr)


if __name__ == "__main__":
    unittest.main()
