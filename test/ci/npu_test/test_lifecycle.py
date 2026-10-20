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

"""Check evidence preservation and cleanup against real filesystem objects."""

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from test.ci.npu_test import lifecycle


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.policy = {
            "repository": "example/tokenspeed",
            "success_seconds": 86400,
            "failure_seconds": 604800,
            "diagnostic_seconds": 1209600,
            "minimum_free_bytes": 0,
            "pending_warning_seconds": 604800,
        }
        for name in ("jobs_root", "evidence_root", "state_root", "diag_root"):
            path = self.root / name
            path.mkdir()
            self.policy[name] = str(path)
        self.now = time.time()
        self.identity = {
            "repository": self.policy["repository"],
            "run_id": "12345",
            "attempt": "1",
            "sha": "a" * 40,
            "ref": "refs/pull/7/merge",
            "runner": "test-runner",
        }
        self.gone = {"pid": 999999999, "boot_id": "missing", "start_ticks": "0"}

    def work(self, owner=None):
        work = lifecycle.initialize(
            self.policy, self.identity, self.gone if owner is None else owner, self.now
        )
        (work / "bootstrap.log").write_text("fatal: connection timed out\n")
        for directory in ("checkout", "cache", "tmp"):
            (work / directory).mkdir()
            (work / directory / "disposable").write_text("temporary")
        phase = work / "accuracy"
        (phase / "example_operator").mkdir(parents=True)
        for name in ("context.json", "report.json"):
            (phase / name).write_text("{}\n")
        (phase / "summary.md").write_text("result\n")
        (phase / "example_operator/accuracy.log").write_text("raw operator output\n")
        return work

    def upload(self, work, accuracy="success", performance="success"):
        bundle = lifecycle.collect(self.policy, work, self.now)
        receipt = lifecycle.acknowledge(
            self.policy,
            work,
            "success",
            "98765",
            "b" * 64,
            accuracy,
            performance,
            self.now,
        )
        self.assertTrue(receipt["uploaded"])
        return bundle

    def test_confirmed_upload_removes_work_and_cache_but_preserves_evidence(self):
        work = self.work()
        bundle = self.upload(work)
        self.assertFalse(work.exists())
        self.assertEqual(
            (bundle / "bootstrap.log").read_text(), "fatal: connection timed out\n"
        )
        self.assertEqual(
            (bundle / "accuracy/example_operator/accuracy.log").read_text(),
            "raw operator output\n",
        )
        self.assertFalse((bundle / "checkout").exists())
        self.assertFalse((bundle / "cache").exists())
        lifecycle.verify_bundle(bundle)

    def test_staging_uses_single_disk_copy(self):
        work = self.work()
        source = work / "accuracy/example_operator/accuracy.log"
        bundle = lifecycle.collect(self.policy, work, self.now)
        self.assertEqual(
            source.stat().st_ino,
            (bundle / "accuracy/example_operator/accuracy.log").stat().st_ino,
        )

    def test_failed_upload_preserves_all_work_and_pending_evidence(self):
        work = self.work()
        bundle = lifecycle.collect(self.policy, work, self.now)
        receipt = lifecycle.acknowledge(
            self.policy, work, "failure", "", "", "failure", "skipped", self.now
        )
        self.assertFalse(receipt["uploaded"])
        result = lifecycle.garbage_collect(self.policy, "apply", self.now + 40 * 86400)
        self.assertTrue((work / "cache/disposable").exists())
        self.assertTrue(bundle.exists())
        self.assertFalse(result["removed"])

    def test_missing_artifact_response_never_releases_work(self):
        work = self.work()
        lifecycle.collect(self.policy, work, self.now)
        receipt = lifecycle.acknowledge(
            self.policy, work, "success", "", "", "success", "success", self.now
        )
        self.assertFalse(receipt["uploaded"])
        self.assertTrue(work.exists())

    def test_modified_evidence_blocks_cleanup(self):
        work = self.work()
        bundle = lifecycle.collect(self.policy, work, self.now)
        (bundle / "bootstrap.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            lifecycle.acknowledge(
                self.policy,
                work,
                "success",
                "1",
                "c" * 64,
                "success",
                "success",
                self.now,
            )
        self.assertTrue(work.exists())

    def test_successful_evidence_expires_after_one_day(self):
        work = self.work()
        bundle = self.upload(work)
        lifecycle.garbage_collect(self.policy, "apply", self.now + 86399)
        self.assertTrue(bundle.exists())
        result = lifecycle.garbage_collect(self.policy, "apply", self.now + 86401)
        self.assertEqual(result["removed"], ["12345-1"])
        self.assertFalse(bundle.exists())

    def test_failed_test_evidence_expires_after_seven_days(self):
        bundle = self.upload(self.work(), "failure", "skipped")
        lifecycle.garbage_collect(self.policy, "apply", self.now + 2 * 86400)
        self.assertTrue(bundle.exists())
        lifecycle.garbage_collect(self.policy, "apply", self.now + 8 * 86400)
        self.assertFalse(bundle.exists())

    def test_dry_run_leaves_eligible_files_intact(self):
        bundle = self.upload(self.work())
        result = lifecycle.garbage_collect(self.policy, "dry-run", self.now + 2 * 86400)
        self.assertEqual(result["eligible"], ["12345-1"])
        self.assertTrue(bundle.exists())

    def test_live_worker_protects_between_phases_and_during_download(self):
        work = self.work(lifecycle.process_identity(os.getpid()))
        bundle = self.upload(work)
        result = lifecycle.garbage_collect(self.policy, "apply", self.now + 40 * 86400)
        self.assertTrue(bundle.exists())
        self.assertIn(
            "active-or-unknown-owner", [row["reason"] for row in result["protected"]]
        )

    def test_legacy_directories_are_not_deleted(self):
        old = Path(self.policy["jobs_root"]) / "actions-99-1.old"
        old.mkdir()
        (old / "old-proof.log").write_text("keep")
        result = lifecycle.garbage_collect(self.policy, "apply", self.now + 40 * 86400)
        self.assertEqual(result["unmanaged"], 1)
        self.assertTrue(old.exists())

    def test_symlink_evidence_is_rejected(self):
        work = self.work()
        target = self.root / "outside-secret"
        target.write_text("never upload")
        (work / "accuracy/example_operator/process.json").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            lifecycle.collect(self.policy, work, self.now)
        self.assertEqual(target.read_text(), "never upload")

    def test_missing_successful_phase_report_fails_collection(self):
        work = self.work()
        lifecycle.phase_state(self.policy, work, "accuracy", "run-phase", 0, self.now)
        (work / "accuracy/report.json").unlink()
        with self.assertRaisesRegex(ValueError, "Required phase evidence"):
            lifecycle.collect(self.policy, work, self.now)

    def test_cannot_delete_work_outside_slot(self):
        outside = self.root / "unrelated-runner"
        outside.mkdir()
        with self.assertRaisesRegex(ValueError, "unsafe deletion"):
            lifecycle.remove_directory(outside, self.policy["jobs_root"])
        self.assertTrue(outside.exists())

    def test_bootstrap_log_has_timestamps_and_preserves_console_annotations(self):
        work = self.work()
        console = io.BytesIO()
        content = b"fatal: transport timeout\n::error::checkout failed\n"
        lifecycle.stream_bootstrap(work, io.BytesIO(content), console)
        self.assertEqual(console.getvalue(), content)
        lines = (work / "bootstrap.log").read_text().splitlines()
        self.assertRegex(lines[-1], r"^\d{4}-\d{2}-\d{2}T.*Z ::error::checkout failed$")

    def test_executor_bootstrap_exception_has_persistent_traceback(self):
        output = self.root / "failed-phase"
        script = Path(lifecycle.__file__).parent / "run_phase.py"
        summary = self.root / "summary.md"
        env = os.environ.copy()
        env.pop("GITHUB_ACTIONS", None)
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(script),
                "--repo-root",
                str(self.root),
                "--phase",
                "accuracy",
                "--output-dir",
                str(output),
                "--timeout-seconds",
                "10",
                "--github-summary",
                str(summary),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("FileNotFoundError", (output / "executor-error.log").read_text())
        self.assertIn("Phase execution failed", summary.read_text())

    def test_old_open_diagnostic_log_is_preserved(self):
        if not Path("/proc").is_dir():
            self.skipTest("Linux /proc open-file protection")
        path = Path(self.policy["diag_root"]) / "Worker_20260101-000000-utc.log"
        path.write_text("active log")
        os.utime(path, (self.now - 40 * 86400,) * 2)
        with path.open():
            lifecycle.garbage_collect(self.policy, "apply", self.now)
            self.assertTrue(path.exists())
        lifecycle.garbage_collect(self.policy, "apply", self.now)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
