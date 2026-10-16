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

"""Latency of the NPU DSA `_scatter` kernel on representative sizes.

Inputs and the output cache are created once and reused. Every sample is timed
with NPU events around a run of kernel launches; JIT compilation and device
warmup happen before timing starts. Only recorded here, no pass/fail threshold.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import traceback

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu._triton import triton
from tokenspeed_kernel_npu.ops.dsa import _scatter

WARMUP = 10
ITERS = 50
SAMPLES = 5


def emit(event, **fields):
    record = {
        "schema_version": 1,
        "operator": "scatter",
        "phase": "performance",
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def launch_scatter(
    x: torch.Tensor, cache: torch.Tensor, locations: torch.Tensor, page: int
) -> None:
    width = x.shape[-1]
    cap = cache.numel() // width
    block = triton.next_power_of_2(width)
    _scatter[(locations.numel(),)](x, cache, locations, width, cap, page, block)


def measure(op, iters: int, samples: int, config_name: str) -> list:
    for _ in range(WARMUP):
        op()
    torch.npu.synchronize()
    per_call_ms = []
    for sample_index in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            op()
        end.record()
        torch.npu.synchronize()
        value = start.elapsed_time(end) / iters
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid NPU timing: {value}")
        per_call_ms.append(value)
        emit("sample", config=config_name, sample_index=sample_index, latency_ms=value)
    return per_call_ms


def run_config(config) -> None:
    emit(
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
    flat = torch.arange(rows * width, dtype=torch.int64) % 201 - 100
    x = flat.to(torch.bfloat16).reshape(rows, width).npu()
    cache = torch.full((cap, width), -99, dtype=torch.bfloat16, device="npu")
    locations = (torch.arange(rows, dtype=torch.int32) + 1).npu()

    def op() -> None:
        launch_scatter(x, cache, locations, page)

    samples = measure(op, ITERS, SAMPLES, config["name"])
    median = statistics.median(samples)
    print(
        f"[scatter-perf] {config['name']}: rows={rows} width={width} "
        f"cache=[{cap},{width}] dtype=bfloat16 iters={ITERS} samples={SAMPLES}"
    )
    print("    per-sample ms: " + ", ".join(f"{value:.4f}" for value in samples))
    print(f"    median latency: {median:.4f} ms/call")
    emit(
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


def configs():
    return [
        {"name": "width64", "width": 64, "rows": 8192, "cap": 16384},
        {"name": "width128_r4096", "width": 128, "rows": 4096, "cap": 8192},
        {"name": "width128_r16384", "width": 128, "rows": 16384, "cap": 32768},
        {"name": "width192", "width": 192, "rows": 8192, "cap": 16384},
        {"name": "width576_r2048", "width": 576, "rows": 2048, "cap": 4096},
        {"name": "width576_r8192", "width": 576, "rows": 8192, "cap": 16384},
    ]


def main() -> int:
    planned = configs()
    completed = 0
    current = None
    emit(
        "started",
        expected_configs=len(planned),
        config_names=[c["name"] for c in planned],
    )
    try:
        if not torch.npu.is_available():
            raise RuntimeError("No NPU device is available")
        torch.npu.set_device(0)
        for config in planned:
            current = config["name"]
            run_config(config)
            completed += 1
    except Exception as exc:  # pragma: no cover - device dependent
        traceback.print_exc()
        emit(
            "failed",
            expected_configs=len(planned),
            completed_configs=completed,
            current_config=current,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"[scatter-perf] FAIL: {exc}", flush=True)
        return 1
    emit("completed", expected_configs=len(planned), completed_configs=completed)
    print("[scatter-perf] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
