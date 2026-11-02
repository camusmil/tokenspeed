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


"""Explicit model smoke and performance interfaces; no undeclared quality baseline."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import qwen_smoke


def main():
    p = argparse.ArgumentParser()
    for key in ["stage", "scenario", "root", "output", "sha", "model", "manifest-sha"]:
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    output = Path(a.output)
    root = Path(a.root)
    profile = json.loads((Path(__file__).parent / "model-profile.json").read_text())
    if profile["topology"]["nodes"] != 1:
        raise ValueError("This registered profile requires one node")
    args = qwen_smoke.parse_args(
        [
            "--source-root",
            str(root),
            "--source-sha",
            a.sha,
            "--install-evidence",
            "/work/runtime/install-evidence.json",
            "--model",
            a.model,
            "--model-id",
            profile["model_id"],
            "--model-revision",
            profile["revision"],
            "--manifest-sha256",
            a.manifest_sha,
            "--dtype",
            profile["dtype"],
            "--sampling-backend",
            "greedy",
            "--temperature",
            "0.0",
            "--device",
            profile["device"],
            "--device-count",
            str(profile["topology"]["devices"]),
            "--tp-size",
            str(profile["topology"]["tp"]),
            "--dp-size",
            str(profile["topology"]["dp"]),
            "--pp-size",
            str(profile["topology"]["pp"]),
            "--prompt",
            profile["prompt"],
            "--max-new-tokens",
            str(profile["max_new_tokens"]),
            "--seed",
            str(profile["seed"]),
            "--port",
            "28888",
            "--output-dir",
            str(output),
        ]
    )
    started = time.monotonic()
    rc = qwen_smoke.worker(args)
    elapsed = time.monotonic() - started
    result = json.loads((output / "result.json").read_text())
    if rc == 0 and a.stage == "model-accuracy" and a.scenario == "model-failure":
        try:
            qwen_smoke.validate_response({"output_ids": [], "meta_info": {}}, 16, 100)
        except ValueError as e:
            result.update(status="failed", injected_failure=str(e))
            rc = 43
    from gates import evaluate

    try:
        gate = evaluate(profile, result, a.stage, elapsed)
    except ValueError as exc:
        gate = {
            "stage": a.stage,
            "execution_status": result["status"],
            "gate_error": str(exc),
            "quality_verdict": "not_evaluated",
            "performance_verdict": "not_evaluated",
        }
        rc = rc or 44
    (output / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
