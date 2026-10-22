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

"""Accuracy and latency of the NPU `virtual_slots_to_local` translation against a CPU reference.

Runs the real Triton kernel through ``tokenspeed_kernel.ops.kvcache`` and checks
the translated integer slots and the boolean owner mask exactly. The reference
mirrors ``_reference_translation`` in
``tokenspeed-kernel/test/ops/test_attention_dsv4_dcp.py``.
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


def emit(phase, event, **fields):
    record = {
        "schema_version": 1,
        "operator": "virtual_slots_to_local",
        "phase": phase,
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def cases():
    planned = []

    def add(rows, degree, rank, dtype, prealloc):
        planned.append(
            {
                "name": f"rows{rows}_deg{degree}_rank{rank}_{dtype}_prealloc{int(prealloc)}",
                "rows": rows,
                "degree": degree,
                "rank": rank,
                "dtype": dtype,
                "geometry": dtype,
                "prealloc": prealloc,
            }
        )

    for rows in (1, 64, 128):
        for degree in (1, 2, 4, 8):
            for rank in range(degree):
                add(rows, degree, rank, "int64", False)
    add(1, 2, 1, "int32", False)
    add(64, 4, 3, "int32", True)
    add(64, 8, 7, "int64", True)
    return planned


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


class VirtualSlotsToLocalTests:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
            virtual_slots_to_local,
        )

        self.torch = torch
        self.virtual_slots_to_local = virtual_slots_to_local

    def reference_translation(self, slots_cpu, rows, virtual_count, degree, rank):
        block = slots_cpu.clamp_min(0) // rows
        owned = (
            (slots_cpu >= rows)
            & (slots_cpu < virtual_count * rows)
            & (((block - 1) % degree) == rank)
        )
        local = ((block - 1) // degree + 1) * rows + slots_cpu.clamp_min(0) % rows
        return self.torch.where(owned, local, self.torch.zeros_like(local)), owned

    def run_case(self, rows, degree, rank, dtype, geometry, use_prealloc) -> bool:
        virtual_count = 1 + 7 * degree * 4
        slots_cpu = self.torch.arange(-5, virtual_count * rows + 9, dtype=dtype)
        slots = slots_cpu.npu()

        expected_local, expected_owned = self.reference_translation(
            slots_cpu.long(), rows, virtual_count, degree, rank
        )

        if use_prealloc:
            out = self.torch.empty_like(
                slots, memory_format=self.torch.contiguous_format
            )
            owner_mask = self.torch.empty(
                slots.shape, dtype=self.torch.bool, device="npu"
            )
            local, owned = self.virtual_slots_to_local(
                slots,
                rows_per_page=rows,
                virtual_block_count=virtual_count,
                degree=degree,
                rank=rank,
                out=out,
                owner_mask=owner_mask,
            )
            reused = local.data_ptr() == out.data_ptr()
            reused = reused and owned.data_ptr() == owner_mask.data_ptr()
            if not reused:
                print("[vsl] FAIL: preallocated outputs were not reused")
                return False
        else:
            local, owned = self.virtual_slots_to_local(
                slots,
                rows_per_page=rows,
                virtual_block_count=virtual_count,
                degree=degree,
                rank=rank,
            )
        self.torch.npu.synchronize()

        actual_local = local.cpu().long()
        actual_owned = owned.cpu()
        if self.torch.equal(actual_local, expected_local) and self.torch.equal(
            actual_owned, expected_owned
        ):
            return True

        print(
            f"[vsl] FAIL: rows={rows} degree={degree} rank={rank} "
            f"dtype={dtype} geometry={geometry} prealloc={use_prealloc}"
        )
        bad_local = (actual_local != expected_local).nonzero().flatten()
        bad_mask = (actual_owned != expected_owned).nonzero().flatten()
        for index in bad_local[:8].tolist():
            print(
                f"    slot={int(slots_cpu[index])} local actual={int(actual_local[index])} "
                f"expected={int(expected_local[index])}"
            )
        for index in bad_mask[:8].tolist():
            print(
                f"    slot={int(slots_cpu[index])} "
                f"owned actual={bool(actual_owned[index])} "
                f"expected={bool(expected_owned[index])}"
            )
        return False

    def measure(self, op, iters: int, samples: int, config_name: str) -> list[float]:
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
        n = config["n"]
        rows = config["rows"]
        degree = config["degree"]
        rank = config["rank"]
        virtual_count = n // rows + 2

        emit(
            "performance",
            "config_started",
            config=config,
            dtype="int64",
            virtual_block_count=virtual_count,
            prealloc=True,
            warmup=WARMUP,
            iterations=ITERS,
            expected_samples=SAMPLES,
        )

        slots = self.torch.arange(n, dtype=self.torch.int64, device="npu")
        out = self.torch.empty_like(slots, memory_format=self.torch.contiguous_format)
        owner_mask = self.torch.empty(slots.shape, dtype=self.torch.bool, device="npu")

        def op() -> None:
            self.virtual_slots_to_local(
                slots,
                rows_per_page=rows,
                virtual_block_count=virtual_count,
                degree=degree,
                rank=rank,
                out=out,
                owner_mask=owner_mask,
            )

        samples = self.measure(op, ITERS, SAMPLES, config["name"])
        median = statistics.median(samples)
        print(
            f"[vsl-perf] {config['name']}: slots={n} rows_per_page={rows} degree={degree} "
            f"rank={rank} dtype=int64 iters={ITERS} samples={SAMPLES}"
        )
        print("    per-sample ms: " + ", ".join(f"{value:.4f}" for value in samples))
        print(f"    median latency: {median:.4f} ms/call")
        emit(
            "performance",
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


OPERATOR = "virtual_slots_to_local"


def accuracy_metadata(case):
    return {"dtype": case["dtype"]}


def performance_metadata(config):
    return {
        "dtype": "int64",
        "prealloc": True,
        "virtual_block_count": config["n"] // config["rows"] + 2,
    }


@pytest.fixture(scope="module")
def npu_suite():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is not installed")
    suite = VirtualSlotsToLocalTests()
    assert suite.torch.npu.is_available(), "No NPU device is available"
    suite.torch.npu.set_device(0)
    return suite


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["name"])
def test_accuracy(case, npu_suite):
    emit("accuracy", "case_started", case=case, **accuracy_metadata(case))
    try:
        ok = npu_suite.run_case(
            case["rows"],
            case["degree"],
            case["rank"],
            getattr(npu_suite.torch, case["dtype"]),
            case["geometry"],
            case["prealloc"],
        )
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
