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

"""Latency of the NPU `virtual_slots_to_local` translation on representative sizes.

Inputs and both output buffers are created once and reused. Every sample is
timed with NPU events around a run of calls; JIT compilation and device warmup
happen before timing starts. Only recorded here, no pass/fail threshold.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import traceback

import torch
import torch_npu  # noqa: F401  (must be imported before the kernel module)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local

WARMUP = 10
ITERS = 50
SAMPLES = 5


def emit(event, **fields):
    record = {
        "schema_version": 1,
        "operator": "virtual_slots_to_local",
        "phase": "performance",
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def measure(op, iters: int, samples: int, config_name: str) -> list[float]:
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
    n = config["n"]
    rows = config["rows"]
    degree = config["degree"]
    rank = config["rank"]
    virtual_count = n // rows + 2

    emit(
        "config_started",
        config=config,
        dtype="int64",
        virtual_block_count=virtual_count,
        prealloc=True,
        warmup=WARMUP,
        iterations=ITERS,
        expected_samples=SAMPLES,
    )

    slots = torch.arange(n, dtype=torch.int64, device="npu")
    out = torch.empty_like(slots, memory_format=torch.contiguous_format)
    owner_mask = torch.empty(slots.shape, dtype=torch.bool, device="npu")

    def op() -> None:
        virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
            out=out,
            owner_mask=owner_mask,
        )

    samples = measure(op, ITERS, SAMPLES, config["name"])
    median = statistics.median(samples)
    print(
        f"[vsl-perf] {config['name']}: slots={n} rows_per_page={rows} degree={degree} "
        f"rank={rank} dtype=int64 iters={ITERS} samples={SAMPLES}"
    )
    print("    per-sample ms: " + ", ".join(f"{value:.4f}" for value in samples))
    print(f"    median latency: {median:.4f} ms/call")
    emit(
        "config_completed",
        config=config,
        dtype="int64",
        virtual_block_count=virtual_count,
        prealloc=True,
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
        {"name": "rows1_deg1", "n": 32768, "rows": 1, "degree": 1, "rank": 0},
        {"name": "rows1_deg8", "n": 65536, "rows": 1, "degree": 8, "rank": 3},
        {"name": "rows1_deg8_large", "n": 131072, "rows": 1, "degree": 8, "rank": 3},
        {"name": "rows64_deg8", "n": 65536, "rows": 64, "degree": 8, "rank": 3},
        {"name": "rows128_deg8", "n": 65536, "rows": 128, "degree": 8, "rank": 3},
        {"name": "rows128_deg4", "n": 131072, "rows": 128, "degree": 4, "rank": 2},
        {
            "name": "rows128_deg8_large",
            "n": 262144,
            "rows": 128,
            "degree": 8,
            "rank": 7,
        },
        {"name": "rows256_deg16", "n": 131072, "rows": 256, "degree": 16, "rank": 5},
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
        print(f"[vsl-perf] FAIL: {exc}", flush=True)
        return 1
    emit("completed", expected_configs=len(planned), completed_configs=completed)
    print("[vsl-perf] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
