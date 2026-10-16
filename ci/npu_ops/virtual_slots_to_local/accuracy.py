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

"""Accuracy of the NPU `virtual_slots_to_local` translation against a CPU reference.

Runs the real Triton kernel through ``tokenspeed_kernel.ops.kvcache`` and checks
the translated integer slots and the boolean owner mask exactly. The reference
mirrors ``_reference_translation`` in
``tokenspeed-kernel/test/ops/test_attention_dsv4_dcp.py``.
"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local


def reference_translation(slots_cpu, rows, virtual_count, degree, rank):
    block = slots_cpu.clamp_min(0) // rows
    owned = (
        (slots_cpu >= rows)
        & (slots_cpu < virtual_count * rows)
        & (((block - 1) % degree) == rank)
    )
    local = ((block - 1) // degree + 1) * rows + slots_cpu.clamp_min(0) % rows
    return torch.where(owned, local, torch.zeros_like(local)), owned


def run_case(rows, degree, rank, dtype, geometry, use_prealloc) -> bool:
    virtual_count = 1 + 7 * degree * 4
    slots_cpu = torch.arange(-5, virtual_count * rows + 9, dtype=dtype)
    slots = slots_cpu.npu()

    expected_local, expected_owned = reference_translation(
        slots_cpu.long(), rows, virtual_count, degree, rank
    )

    if use_prealloc:
        out = torch.empty_like(slots, memory_format=torch.contiguous_format)
        owner_mask = torch.empty(slots.shape, dtype=torch.bool, device="npu")
        local, owned = virtual_slots_to_local(
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
        local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
    torch.npu.synchronize()

    actual_local = local.cpu().long()
    actual_owned = owned.cpu()
    if torch.equal(actual_local, expected_local) and torch.equal(
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


def emit(event, **fields):
    record = {
        "schema_version": 1,
        "operator": "virtual_slots_to_local",
        "phase": "accuracy",
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
            emit("case_started", case=case, dtype=case["dtype"])
            ok = run_case(
                case["rows"],
                case["degree"],
                case["rank"],
                getattr(torch, case["dtype"]),
                case["geometry"],
                case["prealloc"],
            )
            completed += 1
            passed += int(ok)
            emit(
                "case_completed",
                case=case,
                dtype=case["dtype"],
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
        print(f"[vsl] FAIL: {exc}", flush=True)
        return 1
    emit(
        "completed",
        expected_cases=len(planned),
        completed_cases=completed,
        passed_cases=passed,
        failed_cases=completed - passed,
    )
    ok = passed == len(planned)
    print(f"[vsl] PASS ({completed} cases)" if ok else "[vsl] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
