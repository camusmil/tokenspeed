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


"""Evaluate explicitly supplied model baselines without inventing thresholds."""

import math


def evaluate(profile, result, stage, elapsed):
    if stage not in ["model-accuracy", "model-performance"]:
        raise ValueError("registered model stage required")
    if result.get("status") != "passed":
        raise ValueError("model execution failed")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("positive finite elapsed time required")
    ids = result["output_ids"]
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("valid generated tokens required")
    policy = profile["missing_baseline_policy"]
    if policy not in ["not_evaluated", "block"]:
        raise ValueError("unknown missing baseline policy")
    baseline = profile[
        "quality_baseline" if stage == "model-accuracy" else "performance_baseline"
    ]
    gate = {
        "stage": stage,
        "execution_status": "passed",
        "quality_verdict": "not_evaluated",
        "performance_verdict": "not_evaluated",
        "baseline": baseline,
        "elapsed_seconds": elapsed,
        "output_tokens": len(ids),
        "end_to_end_tokens_per_second": len(ids) / elapsed,
    }
    if baseline is None:
        if policy == "block":
            raise ValueError("formal gate requires a frozen baseline")
        gate["reason"] = (
            "No frozen baseline; bounded native smoke and record-only timing"
        )
    elif stage == "model-accuracy":
        if (
            set(baseline) != {"expected_output_ids"}
            or not isinstance(baseline["expected_output_ids"], list)
            or not baseline["expected_output_ids"]
            or any(type(x) is not int or x < 0 for x in baseline["expected_output_ids"])
        ):
            raise ValueError("explicit golden-token baseline required")
        if ids != baseline["expected_output_ids"]:
            raise ValueError("model golden-token baseline failed")
        gate["quality_verdict"] = "passed"
    else:
        if set(baseline) != {"minimum_end_to_end_tokens_per_second"}:
            raise ValueError("explicit end-to-end timing baseline required")
        threshold = baseline["minimum_end_to_end_tokens_per_second"]
        if (
            type(threshold) not in [int, float]
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("positive finite timing threshold required")
        if gate["end_to_end_tokens_per_second"] < threshold:
            raise ValueError("model timing baseline failed")
        gate["performance_verdict"] = "passed"
    return gate
