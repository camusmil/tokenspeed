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

"""Accuracy and latency of the NPU DSA `_scatter` kernel against a CPU reference.

Runs the real Triton kernel from ``tokenspeed_kernel_npu.ops.dsa`` and compares
the whole cache bit-for-bit. The reference follows the kernel contract: a write
lands only when ``PAGE <= loc < CAP``; every other location (null page 0,
negatives, out-of-range) leaves the cache untouched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch

WARMUP = 10
ITERS = 50
SAMPLES = 5
SENTINEL = -99


def emit(phase, event, **fields):
    record = {
        "schema_version": 1,
        "operator": "scatter",
        "phase": phase,
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def cases():
    return [
        {
            "name": "width64",
            "width": 64,
            "cap": 20,
            "page": 1,
            "locations": list(range(1, 13)) + [0, -1, 20, 33],
        },
        {
            "name": "width128",
            "width": 128,
            "cap": 32,
            "page": 1,
            "locations": list(range(1, 17)) + [0, -1, -9, 32, 47],
        },
        {
            "name": "width192",
            "width": 192,
            "cap": 24,
            "page": 1,
            "locations": list(range(1, 13)) + [0, -2, 24, 50],
        },
        {
            "name": "width576",
            "width": 576,
            "cap": 16,
            "page": 1,
            "locations": list(range(1, 11)) + [0, -3, 16, 99],
        },
        {
            "name": "width128_page64",
            "width": 128,
            "cap": 96,
            "page": 64,
            "locations": list(range(64, 80)) + [0, -1, 63, 96, 130],
        },
    ]


def configs():
    return [
        {"name": "width64", "width": 64, "rows": 8192, "cap": 16384},
        {"name": "width128_r4096", "width": 128, "rows": 4096, "cap": 8192},
        {"name": "width128_r16384", "width": 128, "rows": 16384, "cap": 32768},
        {"name": "width192", "width": 192, "rows": 8192, "cap": 16384},
        {"name": "width576_r2048", "width": 576, "rows": 2048, "cap": 4096},
        {"name": "width576_r8192", "width": 576, "rows": 8192, "cap": 16384},
    ]


class ScatterTests:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed_kernel_npu._triton import triton
        from tokenspeed_kernel_npu.ops.dsa import _scatter

        self.torch = torch
        self.triton = triton
        self.scatter = _scatter

    def launch_scatter(
        self, x: torch.Tensor, cache: torch.Tensor, locations: torch.Tensor, page: int
    ) -> None:
        width = x.shape[-1]
        cap = cache.numel() // width
        block = self.triton.next_power_of_2(width)
        self.scatter[(locations.numel(),)](x, cache, locations, width, cap, page, block)

    def reference_cache(self, x_cpu, cache_initial_cpu, locations_cpu, page):
        expected = cache_initial_cpu.clone()
        valid = (locations_cpu >= page) & (locations_cpu < cache_initial_cpu.shape[0])
        expected[locations_cpu[valid]] = x_cpu[valid]
        return expected, valid

    def build_inputs(self, case):
        rows = len(case["locations"])
        width = case["width"]
        flat = self.torch.arange(rows * width, dtype=self.torch.int64) % 201 - 100
        x_cpu = flat.to(self.torch.bfloat16).reshape(rows, width)
        cache_cpu = self.torch.full(
            (case["cap"], width), SENTINEL, dtype=self.torch.bfloat16
        )
        locations_cpu = self.torch.tensor(case["locations"], dtype=self.torch.int64)
        return x_cpu, cache_cpu, locations_cpu

    def run_case(self, case) -> bool:
        name = case["name"]
        x_cpu, cache_cpu, locations_cpu = self.build_inputs(case)

        expected, valid = self.reference_cache(
            x_cpu, cache_cpu, locations_cpu, case["page"]
        )
        targets = locations_cpu[valid]
        if targets.numel() != len(set(targets.tolist())):
            print(f"[scatter] FAIL ({name}): legal target positions are not distinct")
            return False

        x = x_cpu.npu()
        cache = cache_cpu.npu()
        locations = locations_cpu.to(self.torch.int32).npu()
        self.launch_scatter(x, cache, locations, case["page"])
        self.torch.npu.synchronize()

        actual = cache.cpu()
        if self.torch.equal(actual, expected):
            print(
                f"[scatter] {name}: PASS  (width={case['width']}, cap={case['cap']}, "
                f"page={case['page']}, rows={len(case['locations'])})"
            )
            return True

        diff = (actual != expected).nonzero()
        print(f"[scatter] FAIL ({name}): {diff.shape[0]} mismatching elements")
        for idx in diff[:8].tolist():
            row, col = idx
            got = float(actual[row, col])
            want = float(expected[row, col])
            print(f"    cache[{row},{col}] actual={got} expected={want}")
        return False

    def measure(self, op, iters: int, samples: int, config_name: str) -> list:
        for _ in range(WARMUP):
            op()
        self.torch.npu.synchronize()
        per_call_ms = []
        for sample_index in range(samples):
            start = self.torch.npu.Event(enable_timing=True)
            end = self.torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                op()
            end.record()
            self.torch.npu.synchronize()
            value = start.elapsed_time(end) / iters
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid NPU timing: {value}")
            per_call_ms.append(value)
            emit(
                "performance",
                "sample",
                config=config_name,
                sample_index=sample_index,
                latency_ms=value,
            )
        return per_call_ms

    def run_config(self, config) -> None:
        emit(
            "performance",
            "config_started",
            config=config,
            dtype="bfloat16",
            page=1,
            warmup=WARMUP,
            iterations=ITERS,
            expected_samples=SAMPLES,
        )
        rows = config["rows"]
        width = config["width"]
        cap = config["cap"]
        page = 1
        flat = self.torch.arange(rows * width, dtype=self.torch.int64) % 201 - 100
        x = flat.to(self.torch.bfloat16).reshape(rows, width).npu()
        cache = self.torch.full(
            (cap, width), -99, dtype=self.torch.bfloat16, device="npu"
        )
        locations = (self.torch.arange(rows, dtype=self.torch.int32) + 1).npu()

        def op() -> None:
            self.launch_scatter(x, cache, locations, page)

        samples = self.measure(op, ITERS, SAMPLES, config["name"])
        median = statistics.median(samples)
        print(
            f"[scatter-perf] {config['name']}: rows={rows} width={width} "
            f"cache=[{cap},{width}] dtype=bfloat16 iters={ITERS} samples={SAMPLES}"
        )
        print("    per-sample ms: " + ", ".join(f"{value:.4f}" for value in samples))
        print(f"    median latency: {median:.4f} ms/call")
        emit(
            "performance",
            "config_completed",
            config=config,
            dtype="bfloat16",
            page=page,
            warmup=WARMUP,
            iterations=ITERS,
            sample_count=len(samples),
            samples_ms=samples,
            median_latency_ms=median,
            threshold_ms=None,
            performance_verdict="not_evaluated",
        )


OPERATOR = "scatter"


def accuracy_metadata(case):
    return {"dtype": "bfloat16"}


def performance_metadata(config):
    return {"dtype": "bfloat16", "page": 1}


@pytest.fixture(scope="module")
def npu_suite():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is not installed")
    suite = ScatterTests()
    assert suite.torch.npu.is_available(), "No NPU device is available"
    suite.torch.npu.set_device(0)
    return suite


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["name"])
def test_accuracy(case, npu_suite):
    emit("accuracy", "case_started", case=case, **accuracy_metadata(case))
    try:
        ok = npu_suite.run_case(case)
    except Exception:
        emit(
            "accuracy",
            "case_completed",
            case=case,
            status="failed",
            **accuracy_metadata(case),
        )
        raise
    emit(
        "accuracy",
        "case_completed",
        case=case,
        status="passed" if ok else "failed",
        **accuracy_metadata(case),
    )
    assert ok, f"Accuracy mismatch in case {case['name']}"


@pytest.mark.parametrize("config", configs(), ids=lambda config: config["name"])
def test_performance(config, npu_suite):
    npu_suite.run_config(config)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "test/ci_system/npu"))
    from lifecycle import test_entry

    raise SystemExit(test_entry(Path(__file__)))
