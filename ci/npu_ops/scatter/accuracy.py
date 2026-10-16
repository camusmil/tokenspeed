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

"""Accuracy of the NPU DSA `_scatter` kernel against a CPU reference.

Runs the real Triton kernel from ``tokenspeed_kernel_npu.ops.dsa`` and compares
the whole cache bit-for-bit. The reference follows the kernel contract: a write
lands only when ``PAGE <= loc < CAP``; every other location (null page 0,
negatives, out-of-range) leaves the cache untouched.
"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu._triton import triton
from tokenspeed_kernel_npu.ops.dsa import _scatter

SENTINEL = -99


def launch_scatter(
    x: torch.Tensor, cache: torch.Tensor, locations: torch.Tensor, page: int
) -> None:
    width = x.shape[-1]
    cap = cache.numel() // width
    block = triton.next_power_of_2(width)
    _scatter[(locations.numel(),)](x, cache, locations, width, cap, page, block)


def reference_cache(x_cpu, cache_initial_cpu, locations_cpu, page):
    expected = cache_initial_cpu.clone()
    valid = (locations_cpu >= page) & (locations_cpu < cache_initial_cpu.shape[0])
    expected[locations_cpu[valid]] = x_cpu[valid]
    return expected, valid


def build_inputs(case):
    rows = len(case["locations"])
    width = case["width"]
    flat = torch.arange(rows * width, dtype=torch.int64) % 201 - 100
    x_cpu = flat.to(torch.bfloat16).reshape(rows, width)
    cache_cpu = torch.full((case["cap"], width), SENTINEL, dtype=torch.bfloat16)
    locations_cpu = torch.tensor(case["locations"], dtype=torch.int64)
    return x_cpu, cache_cpu, locations_cpu


def run_case(case) -> bool:
    name = case["name"]
    x_cpu, cache_cpu, locations_cpu = build_inputs(case)

    expected, valid = reference_cache(x_cpu, cache_cpu, locations_cpu, case["page"])
    targets = locations_cpu[valid]
    if targets.numel() != len(set(targets.tolist())):
        print(f"[scatter] FAIL ({name}): legal target positions are not distinct")
        return False

    x = x_cpu.npu()
    cache = cache_cpu.npu()
    locations = locations_cpu.to(torch.int32).npu()
    launch_scatter(x, cache, locations, case["page"])
    torch.npu.synchronize()

    actual = cache.cpu()
    if torch.equal(actual, expected):
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


def emit(event, **fields):
    record = {
        "schema_version": 1,
        "operator": "scatter",
        "phase": "accuracy",
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def main() -> int:
    planned = cases()
    completed = 0
    passed = 0
    current = None
    emit(
        "started", expected_cases=len(planned), case_names=[c["name"] for c in planned]
    )
    try:
        if not torch.npu.is_available():
            raise RuntimeError("No NPU device is available")
        torch.npu.set_device(0)
        for case in planned:
            current = case["name"]
            emit("case_started", case=case, dtype="bfloat16")
            ok = run_case(case)
            completed += 1
            passed += int(ok)
            emit(
                "case_completed",
                case=case,
                dtype="bfloat16",
                status="passed" if ok else "failed",
            )
            if not ok:
                raise AssertionError(f"Accuracy mismatch in case {current}")
    except Exception as exc:  # pragma: no cover - device dependent
        traceback.print_exc()
        emit(
            "failed",
            expected_cases=len(planned),
            completed_cases=completed,
            passed_cases=passed,
            failed_cases=completed - passed,
            current_case=current,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"[scatter] FAIL: {exc}", flush=True)
        return 1
    emit(
        "completed",
        expected_cases=len(planned),
        completed_cases=completed,
        passed_cases=passed,
        failed_cases=completed - passed,
    )
    ok = passed == len(planned)
    print("[scatter] PASS" if ok else "[scatter] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
