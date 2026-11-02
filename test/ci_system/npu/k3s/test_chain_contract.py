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


import importlib.util
import json
import runpy
import tempfile
import unittest
from pathlib import Path


class ChainContractTests(unittest.TestCase):
    def test_current_operator_plan_and_counts(self):
        root = Path(__file__).resolve().parents[4]
        api = runpy.run_path(
            str(root / "test/ci_system/npu/run_phase.py"), run_name="cp12_plan_test"
        )
        _, manifest, plan, _ = api["load_inputs"](root)
        self.assertEqual(
            sum(len(x["accuracy"]) for x in plan["operators"].values()),
            sum(x["accuracy_cases"] for x in manifest["operators"]),
        )
        self.assertEqual(
            sum(len(x["performance"]) for x in plan["operators"].values()),
            sum(x["performance_configs"] for x in manifest["operators"]),
        )

    def test_changed_script_rejected_by_original_plan(self):
        root = Path(__file__).resolve().parents[4]
        api = runpy.run_path(
            str(root / "test/ci_system/npu/run_phase.py"), run_name="cp12_plan_test"
        )
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for relative in ["test/ci/npu", "test/ci_system/npu"]:
                shutil.copytree(root / relative, target / relative)
            with (target / "test/ci/npu/tests/scatter-test.py").open("a") as f:
                f.write("\n# altered after plan registration\n")
            with self.assertRaises(ValueError):
                api["load_inputs"](target)

    def test_model_baselines_fail_closed_and_remain_explicit(self):
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from gates import evaluate

        profile = json.loads((Path(__file__).parent / "model-profile.json").read_text())
        result = {"status": "passed", "output_ids": [1, 2]}
        self.assertEqual(
            evaluate(profile, result, "model-accuracy", 2)["quality_verdict"],
            "not_evaluated",
        )
        self.assertEqual(
            evaluate(profile, result, "model-performance", 2)[
                "end_to_end_tokens_per_second"
            ],
            1,
        )
        with self.assertRaises(ValueError):
            evaluate(
                dict(profile, missing_baseline_policy="block"),
                result,
                "model-accuracy",
                2,
            )
        self.assertEqual(
            evaluate(
                dict(profile, quality_baseline={"expected_output_ids": [1, 2]}),
                result,
                "model-accuracy",
                2,
            )["quality_verdict"],
            "passed",
        )
        with self.assertRaises(ValueError):
            evaluate(
                dict(profile, quality_baseline={"expected_output_ids": [3]}),
                result,
                "model-accuracy",
                2,
            )
        with self.assertRaises(ValueError):
            evaluate(
                dict(
                    profile,
                    performance_baseline={"minimum_end_to_end_tokens_per_second": 2},
                ),
                result,
                "model-performance",
                2,
            )
        with self.assertRaises(ValueError):
            evaluate(
                profile, {"status": "failed", "output_ids": [1]}, "model-accuracy", 1
            )


if __name__ == "__main__":
    unittest.main()
