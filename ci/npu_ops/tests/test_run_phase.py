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

"""CPU regressions using actual subprocesses and the structured report reader."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_phase import capture_test, load_inputs


class PhaseTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.source = Path(__file__).resolve().parents[3]
        self.temp = None
        self.root = None
        self.output = None
        self.manifest = None
        self.plan = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        target = self.root / "ci/npu_ops"
        target.mkdir(parents=True)
        for name in ("changes.py", "reporting.py", "run_phase.py"):
            shutil.copyfile(self.source / "ci/npu_ops" / name, target / name)
        for name in ("tokenspeed-kernel/python", "tokenspeed-kernel-npu/python"):
            (self.root / name).mkdir(parents=True)
        self.manifest = json.loads(
            (self.source / "ci/npu_ops/manifest.json").read_text()
        )
        self.manifest["operators"] = []
        self.plan = {"schema_version": 2, "operators": {}, "source_sha256": {}}
        for name in ("alpha", "beta"):
            scripts = {
                phase: f"ci/npu_ops/{name}/{phase}.py"
                for phase in ("accuracy", "performance")
            }
            self.manifest["operators"].append(
                dict(
                    name=name, scripts=scripts, accuracy_cases=1, performance_configs=1
                )
            )
            self.plan["operators"][name] = {
                "accuracy": [
                    {"case": {"name": "one"}, "metadata": {"dtype": "float32"}}
                ],
                "performance": [
                    {"config": {"name": "one"}, "metadata": {"dtype": "float32"}}
                ],
                "timing": {"warmup": 0, "iterations": 1, "sample_count": 1},
            }
            for phase, path in scripts.items():
                records = self.records(name, phase)
                script = self.root / path
                script.parent.mkdir(exist_ok=True)
                script.write_text(
                    "import json\n"
                    + "\n".join(
                        "print('NPU_RESULT_JSON=' + json.dumps("
                        + repr(r)
                        + "), flush=True)"
                        for r in records
                    )
                    + "\n"
                )
        self.sync_plan()

    def records(self, name, phase):
        def event(kind, **fields):
            return dict(
                schema_version=1, operator=name, phase=phase, event=kind, **fields
            )

        if phase == "accuracy":
            return [
                event("started", expected_cases=1, case_names=["one"]),
                event("case_started", case={"name": "one"}, dtype="float32"),
                event(
                    "case_completed",
                    case={"name": "one"},
                    dtype="float32",
                    status="passed",
                ),
                event(
                    "completed",
                    expected_cases=1,
                    completed_cases=1,
                    passed_cases=1,
                    failed_cases=0,
                ),
            ]
        config = dict(config={"name": "one"}, dtype="float32", warmup=0, iterations=1)
        return [
            event("started", expected_configs=1, config_names=["one"]),
            event("config_started", **config, expected_samples=1),
            event("sample", config="one", sample_index=0, latency_ms=0.5),
            event(
                "config_completed",
                **config,
                sample_count=1,
                samples_ms=[0.5],
                median_latency_ms=0.5,
                threshold_ms=None,
                performance_verdict="not_evaluated",
            ),
            event("completed", expected_configs=1, completed_configs=1),
        ]

    def sync_plan(self):
        path = self.root / "ci/npu_ops/manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.plan["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for entry in self.manifest["operators"]:
            for path in entry["scripts"].values():
                self.plan["source_sha256"][path] = hashlib.sha256(
                    (self.root / path).read_bytes()
                ).hexdigest()
        (self.root / "ci/npu_ops/expected-plans.json").write_text(json.dumps(self.plan))

    def append(self, phase, code):
        path = self.root / f"ci/npu_ops/alpha/{phase}.py"
        path.write_text(path.read_text() + code + "\n")
        self.sync_plan()

    def run_phase(self, phase, summary):
        env = dict(os.environ, GITHUB_ACTIONS="true")
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-u",
                str(self.root / "ci/npu_ops/run_phase.py"),
                "--repo-root",
                str(self.root),
                "--phase",
                phase,
                "--output-dir",
                str(self.output),
                "--timeout-seconds",
                "1",
                "--github-summary",
                str(summary),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )

    def test_success_both_phases_use_report_validation(self):
        for phase in ("accuracy", "performance"):
            self.output = self.root / phase
            result = self.run_phase(phase, self.root / "summary.md")
            self.assertEqual(result.returncode, 0, result.stdout)
            report = json.loads((self.output / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertIn("Execution finished", result.stdout)
            summary = (self.output / "summary.md").read_text()
            self.assertIn(f"## Ascend NPU operator {phase} tests", summary)
            self.assertIn("Phase status: **passed**", summary)
            self.assertIn("| Operator | Result | Progress | Exit code |", summary)
            self.assertIn("### Evidence", summary)
            self.assertIn("### Run information", summary)
            self.assertTrue(summary.isascii(), summary)
            self.assertIn(summary, (self.root / "summary.md").read_text())
            if phase == "performance":
                self.assertEqual(report["performance_verdict"], "not_evaluated")
                self.assertIn("Performance verdict: **not_evaluated**", summary)
                self.assertIn("no latency thresholds are configured", summary)
                self.assertIn("### Configuration sampling results", summary)
                self.assertIn("Median latency (ms/execution)", summary)
            else:
                self.assertIn(
                    "Completed 1/1; passed 1; failed 0; incomplete 0; not run 0",
                    summary,
                )

    def test_nonzero_exit_stops_later_operator_in_both_phases(self):
        for phase in ("accuracy", "performance"):
            self.output = self.root / phase
            self.append(phase, "raise SystemExit(7)")
            result = self.run_phase(phase, self.root / "summary.md")
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((self.output / "report.json").read_text())
            self.assertEqual(report["operators"][0]["exit_code"], 7)
            self.assertEqual(report["operators"][1]["status"], "not_run")
            self.assertFalse((self.output / "beta").exists())
            self.assertIn("::error::", result.stdout)
            summary = (self.output / "summary.md").read_text()
            self.assertIn("Phase status: **failed**", summary)
            self.assertIn("### Failure and execution details", summary)
            self.assertIn("beta: not run after failure of alpha", summary)

    def test_timeout_stops_later_operator(self):
        self.append("accuracy", "import time; time.sleep(60)")
        result = self.run_phase("accuracy", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["operators"][0]["error_type"], "TimeoutError")
        self.assertEqual(report["operators"][1]["status"], "not_run")

    def test_operator_evidence_failure_stops_later_operator(self):
        self.append(
            "accuracy",
            f"from pathlib import Path; Path({str(self.output / 'alpha/process.json')!r}).mkdir()",
        )
        result = self.run_phase("accuracy", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertIn("evidence_error", report["operators"][0])
        self.assertEqual(report["operators"][1]["status"], "not_run")

    def test_report_write_failure_propagates(self):
        self.append(
            "accuracy",
            f"from pathlib import Path; Path({str(self.output / 'report.json')!r}).mkdir()",
        )
        result = self.run_phase("accuracy", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("report.json", result.stdout)
        self.assertIn("Phase execution failed", result.stdout)
        summary = (self.root / "summary.md").read_text()
        self.assertIn("### ARM NPU phase execution failed", summary)
        self.assertIn("Phase: `accuracy`. Remaining tests have been stopped.", summary)
        self.assertIn("Phase execution failed", summary)

    def test_actions_summary_failure_propagates(self):
        result = self.run_phase("accuracy", self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to save summary", result.stdout)

    def test_hash_failure_prevents_tests_and_emits_escaped_annotation(self):
        path = self.root / "ci/npu_ops/alpha/accuracy.py"
        path.write_text("raise AssertionError('must not run')\n")
        result = self.run_phase("accuracy", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Script differs from expected plan", result.stdout)
        self.assertFalse(self.output.exists())

    def test_failure_annotation_escapes_original_error(self):
        script = self.root / "ci/npu_ops/alpha/accuracy.py"
        records = self.records("alpha", "accuracy")[:2]
        records.append(
            dict(
                schema_version=1,
                operator="alpha",
                phase="accuracy",
                event="failed",
                expected_cases=1,
                completed_cases=0,
                passed_cases=0,
                failed_cases=0,
                current_case="one",
                error_type="RuntimeError",
                error="bad%value\nnext\rline \u539f\u59cb\u9519\u8bef",
            )
        )
        script.write_text(
            "import json\n"
            + "\n".join(
                "print('NPU_RESULT_JSON=' + json.dumps(" + repr(r) + "))"
                for r in records
            )
            + "\nraise SystemExit(3)\n"
        )
        self.sync_plan()
        result = self.run_phase("accuracy", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "::error::alpha/accuracy: bad%25value%0Anext%0Dline \u539f\u59cb\u9519\u8bef",
            result.stdout,
        )
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["operators"][0]["error"], records[-1]["error"])
        self.assertIn(
            "\u539f\u59cb\u9519\u8bef", (self.output / "summary.md").read_text()
        )

    def test_missing_sample_rejected(self):
        path = self.root / "ci/npu_ops/alpha/performance.py"
        path.write_text(
            "\n".join(
                line
                for line in path.read_text().splitlines()
                if "'event': 'sample'" not in line
            )
            + "\n"
        )
        self.sync_plan()
        result = self.run_phase("performance", self.root / "summary.md")
        self.assertNotEqual(result.returncode, 0)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["operators"][1]["status"], "not_run")

    def test_log_open_failure_does_not_launch_command(self):
        marker = self.root / "unexpected"
        result = capture_test(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
            self.root,
            self.root,
            1,
        )
        self.assertFalse(result["log_saved"])
        self.assertIsNotNone(result["capture_error"])
        self.assertFalse(marker.exists())

    def test_timeout_kills_descendant_process(self):
        marker = self.root / "orphan"
        child = f"import time; time.sleep(1); open({str(marker)!r}, 'w').close()"
        parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
        result = capture_test(
            [sys.executable, "-u", "-c", parent], self.root, self.root / "raw.log", 0.2
        )
        self.assertEqual(result["capture_error"]["type"], "TimeoutError")
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def test_architecture_contract_rejects_x86_and_non_linux(self):
        for system, architecture, expected in (
            ("Linux", "aarch64", 0),
            ("Linux", "arm64", 0),
            ("Linux", "x86_64", 1),
            ("Darwin", "arm64", 1),
        ):
            result = subprocess.run(
                [
                    "bash",
                    str(self.source / "ci/npu_ops/run_step.sh"),
                    "--check-platform",
                    system,
                    architecture,
                ],
                capture_output=True,
            )
            self.assertEqual(result.returncode, expected)

    def test_repository_manifest_and_plan_are_consistent(self):
        _, manifest, plan, _ = load_inputs(self.source)
        self.assertEqual(
            {entry["name"] for entry in manifest["operators"]}, set(plan["operators"])
        )
        for entry in manifest["operators"]:
            operator_plan = plan["operators"][entry["name"]]
            self.assertEqual(len(operator_plan["accuracy"]), entry["accuracy_cases"])
            self.assertEqual(
                len(operator_plan["performance"]), entry["performance_configs"]
            )
            self.assertGreater(operator_plan["timing"]["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
