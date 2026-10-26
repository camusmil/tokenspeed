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

"""Generate one bounded native-NPU response from an offline pinned Qwen model.

The NPU pipeline owns process supervision, locking, deadlines and archiving.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="runtime-1npu")


def positive_seconds(value):
    value = int(value)
    if value <= 0:
        raise ValueError("Expected positive integer")
    return value


def save_text(path, text):
    path = Path(path)
    with path.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def write_json(path, value):
    save_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def check_model(model, model_id, revision, manifest_sha256):
    if (
        model != model.resolve(strict=True)
        or model.stat().st_uid != 0
        or model.stat().st_mode & 0o222
    ):
        raise ValueError(
            "Model directory must be an absolute root-owned readonly directory"
        )
    path = model / "model-manifest.json"
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != 0
        or path.stat().st_mode & 0o222
        or path.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Invalid model manifest")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Model manifest digest differs from the selected task")
    manifest = json.loads(raw)
    if manifest["model_id"] != model_id or manifest["revision"] != revision:
        raise ValueError("Model identity differs from the selected task")
    wanted = {"model-manifest.json"}
    for entry in manifest["files"]:
        path = model / entry["path"]
        if (
            path.parent != model
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_uid != 0
            or path.stat().st_mode & 0o222
        ):
            raise ValueError(f"Unsafe or writable model file: {entry['path']}")
        if entry["path"] in wanted:
            raise ValueError("Duplicate model manifest entry")
        if (
            path.stat().st_size != entry["size"]
            or hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            != entry["sha256"]
        ):
            raise ValueError(f"Model content mismatch: {entry['path']}")
        wanted.add(entry["path"])
    if {p.name for p in model.iterdir()} != wanted:
        raise ValueError("Unexpected or missing model files")
    if not {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }.issubset(wanted):
        raise ValueError("Incomplete offline model")
    return manifest_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--install-evidence", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dtype", choices=("bfloat16",), required=True)
    parser.add_argument("--sampling-backend", choices=("greedy",), required=True)
    parser.add_argument("--temperature", type=float, choices=(0.0,), required=True)
    parser.add_argument("--device", choices=("npu",), required=True)
    parser.add_argument("--device-count", type=int, choices=(1,), required=True)
    parser.add_argument("--tp-size", type=int, choices=(1,), required=True)
    parser.add_argument("--dp-size", type=int, choices=(1,), required=True)
    parser.add_argument("--pp-size", type=int, choices=(1,), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=positive_seconds, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch("[0-9a-f]{40}", args.source_sha):
        parser.error("source SHA must be a full lowercase Git SHA")
    if not re.fullmatch("[0-9a-f]{40}", args.model_revision) or not re.fullmatch(
        "[0-9a-f]{64}", args.manifest_sha256
    ):
        parser.error("model revision and manifest digest must be full hashes")
    if not args.model_id.strip() or not args.prompt.strip():
        parser.error("model-id and prompt must not be empty")
    if not 1 <= args.max_new_tokens <= 64 or not 1024 <= args.port <= 65535:
        parser.error("smoke requires 1..64 generated tokens and port 1024..65535")
    return args


def check_runtime_binary(binary, prefix):
    path = Path(binary["path"]).resolve(strict=True)
    if not path.is_relative_to(prefix.resolve(strict=True)):
        raise ValueError("Runtime binary is outside this task's virtualenv")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != binary["sha256"]:
            raise ValueError("Installed runtime binary identity mismatch")


def check_install_evidence(path, source_sha):
    evidence = json.loads(path.read_text())
    if (
        evidence.get("components") != "full-runtime"
        or evidence.get("source_sha") != source_sha
        or evidence.get("pip_check_passed") is not True
    ):
        raise ValueError(
            "Current-source full-runtime installation and pip check evidence required"
        )
    actual = subprocess.check_output(
        ["git", "-C", evidence["source_root"], "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source_sha:
        raise ValueError("Installation source commit mismatch")
    subprocess.run(
        ["git", "-C", evidence["source_root"], "diff", "--exit-code", "HEAD", "--"],
        check=True,
    )
    expected = {
        "tokenspeed": "python",
        "tokenspeed-scheduler": "tokenspeed-scheduler",
        "tokenspeed-engine": "tokenspeed-engine",
        "tokenspeed-kernel": "tokenspeed-kernel/python",
        "tokenspeed-kernel-npu": "tokenspeed-kernel-npu",
    }
    if set(evidence["installed"]) != set(expected):
        raise ValueError("Missing installed runtime components")
    source = Path(evidence["source_root"]).resolve(strict=True)
    for name, item in evidence["installed"].items():
        distribution = importlib.metadata.distribution(name)
        actual_url = json.loads(distribution.read_text("direct_url.json"))
        if (
            item["version"] != distribution.version
            or actual_url != item["direct_url"]
            or actual_url["url"] != (source / expected[name]).as_uri()
        ):
            raise ValueError(f"Installed source provenance mismatch: {name}")
        if name == "tokenspeed-scheduler" and "binary" not in item:
            raise ValueError("Missing scheduler binary evidence")
        binaries = [item["binary"]] if "binary" in item else []
        if name == "tokenspeed-engine":
            if set(item.get("binaries", {})) != {"native", "host"}:
                raise ValueError("Missing engine binary evidence")
            binaries.extend(item["binaries"].values())
        for binary in binaries:
            check_runtime_binary(binary, Path(sys.prefix))
    return evidence


def require_source_path(filename, source_directory):
    actual = Path(filename).resolve(strict=True)
    expected = source_directory.resolve(strict=True)
    if not actual.is_relative_to(expected):
        raise RuntimeError(
            f"Import is outside this job source: {actual}; expected {expected}"
        )
    return str(actual)


def validate_response(response, maximum, vocabulary_size):
    if not isinstance(response, dict):
        raise ValueError("Generation did not return an object")
    ids = response.get("output_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= maximum:
        raise ValueError("Generation token count is outside the smoke contract")
    if any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
        raise ValueError("Generation returned invalid token IDs")
    meta = response.get("meta_info")
    if not isinstance(meta, dict) or not isinstance(meta.get("finish_reason"), dict):
        raise ValueError("Generation did not return a finish reason")
    finish = meta["finish_reason"].get("type")
    if finish not in ("length", "stop"):
        raise ValueError(f"Unexpected finish reason: {finish!r}")
    if type(meta.get("completion_tokens")) is not int or meta[
        "completion_tokens"
    ] != len(ids):
        raise ValueError("Completion token evidence is inconsistent")
    json.dumps(response, allow_nan=False)
    return ids


def worker(args):
    result = {
        "schema_version": 1,
        "status": "failed",
        "source_sha": args.source_sha,
        "model": str(args.model),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "started_at": time.time(),
        "stage": "preflight",
    }
    engine = None
    exit_code = 1
    try:
        if os.environ.get("TOKENSPEED_CI_SOURCE_SHA") != args.source_sha:
            raise RuntimeError("Source SHA differs from the VM task provenance")
        if os.environ.get("TOKENSPEED_KERNEL_BACKEND") != "npu":
            raise RuntimeError("TOKENSPEED_KERNEL_BACKEND=npu is required")
        result["install"] = check_install_evidence(
            args.install_evidence, args.source_sha
        )
        if (
            Path(result["install"]["source_root"]).resolve()
            != args.source_root.resolve()
        ):
            raise ValueError("Installed source root differs from the selected checkout")
        # Unset before the first torch import, including imports in subprocesses.
        os.environ.pop("TORCH_TRANSFER_TO_NPU", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        directories = [
            args.source_root / relative
            for relative in (
                "python",
                "tokenspeed-kernel/python",
                "tokenspeed-kernel-npu/python",
            )
        ]
        for directory in directories:
            directory.resolve(strict=True)
        sys.path[:0] = [str(path) for path in directories]
        os.environ["PYTHONPATH"] = os.pathsep.join(str(path) for path in directories)
        checked = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        save_text(args.output_dir / "pip-check.log", checked.stdout + checked.stderr)
        if checked.returncode:
            raise RuntimeError(f"pip check failed: {checked.returncode}")
        from packaging.version import Version

        if Version(importlib.metadata.version("tokenspeed-scheduler")) < Version(
            "0.1.40"
        ):
            raise RuntimeError("Current runtime requires tokenspeed-scheduler>=0.1.40")
        result["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torch-npu",
                "triton-ascend",
                "transformers",
                "protobuf",
                "tokenspeed-scheduler",
                "tokenspeed-engine",
                "tokenspeed-mooncake",
            )
        }
        result["stage"] = "native_npu"
        import torch
        import torch_npu  # noqa: F401

        if "torch_npu.contrib.transfer_to_npu" in sys.modules:
            raise RuntimeError("Automatic CUDA-to-NPU migration must not be loaded")
        if (
            not torch.npu.is_available()
            or torch.npu.device_count() != args.device_count
        ):
            raise RuntimeError(
                "Native NPU visibility differs from the requested device count"
            )
        probe = torch.arange(8, device=args.device, dtype=torch.float32) + 1
        torch.npu.synchronize()
        if probe.device.type != "npu" or not torch.equal(
            probe.cpu(), torch.arange(1, 9, dtype=torch.float32)
        ):
            raise RuntimeError("Native NPU probe failed")
        result["native_npu"] = {
            "visible_devices": torch.npu.device_count(),
            "probe_values": probe.cpu().tolist(),
            "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "automatic_cuda_migration_loaded": False,
        }
        result["imports"] = {}
        for name, directory in zip(
            ("tokenspeed", "tokenspeed_kernel", "tokenspeed_kernel_npu"), directories
        ):
            module = importlib.import_module(name)
            result["imports"][name] = require_source_path(module.__file__, directory)
        extension = importlib.import_module(
            "tokenspeed_scheduler.tokenspeed_scheduler_ext"
        )
        result["imports"]["tokenspeed_scheduler_ext"] = require_source_path(
            extension.__file__, Path(sys.prefix)
        )
        if (
            Path(extension.__file__).resolve()
            != Path(
                result["install"]["installed"]["tokenspeed-scheduler"]["binary"]["path"]
            ).resolve()
        ):
            raise RuntimeError(
                "Loaded scheduler binary differs from installation evidence"
            )
        import tokenspeed_engine
        import tokenspeed_engine._native as engine_native

        for label, filename in {
            "native": engine_native.__file__,
            "host": tokenspeed_engine.host_binary(),
        }.items():
            actual = require_source_path(filename, Path(sys.prefix))
            if (
                actual
                != result["install"]["installed"]["tokenspeed-engine"]["binaries"][
                    label
                ]["path"]
            ):
                raise ValueError(
                    "Loaded engine binary differs from installation evidence"
                )
            result["imports"][f"engine_{label}"] = actual
        model = args.model.resolve(strict=True)
        result["model_manifest_sha256"] = check_model(
            model, args.model_id, args.model_revision, args.manifest_sha256
        )
        config = model / "config.json"
        if json.loads(config.read_text()).get("model_type") != "qwen3":
            raise ValueError("This smoke supports the reviewed Qwen3 model only")
        result["model_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model), local_files_only=True, trust_remote_code=False
        )
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        from tokenspeed.runtime.entrypoints.engine import Engine

        result["imports"]["engine"] = require_source_path(
            sys.modules[Engine.__module__].__file__, args.source_root / "python"
        )
        engine_args = dict(
            model=str(model),
            tokenizer=str(model),
            served_model_name="qwen-npu-smoke",
            device=args.device,
            dtype=args.dtype,
            kv_cache_dtype="auto",
            trust_remote_code=False,
            world_size=args.device_count,
            attn_tp_size=args.tp_size,
            data_parallel_size=args.dp_size,
            pipeline_parallel_size=args.pp_size,
            attention_backend="mha",
            sampling_backend=args.sampling_backend,
            grammar_backend="none",
            disable_prefill_graph=True,
            disable_pdl=True,
            disable_autotune=True,
            enforce_eager=False,
            max_model_len=4096,
            max_num_seqs=1,
            max_total_tokens=16384,
            chunked_prefill_size=4096,
            prefix_granularity=128,
            max_cudagraph_capture_size=1,
            cudagraph_capture_sizes=[1],
            host="127.0.0.1",
            port=args.port,
            seed=args.seed,
            log_level="info",
        )
        result["engine_args"] = engine_args
        import socket

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", args.port))
        result["stage"] = "engine_create"
        engine = Engine(**engine_args)
        if engine.server_args.device != "npu":
            raise RuntimeError("Engine did not select the native NPU device")
        result["stage"] = "generate"
        response = engine.generate(
            input_ids=list(input_ids),
            sampling_params={
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            stream=False,
        )
        ids = validate_response(response, args.max_new_tokens, len(tokenizer))
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if not decoded.strip():
            raise ValueError("Generation decoded to empty output")
        result.update(response=response, output_ids=ids, decoded_output=decoded)
        # Confirm JSON evidence contains no NaN/Infinity before declaring success.
        json.dumps(result, allow_nan=False)
        result["status"] = "passed"
        result["stage"] = "completed"
        exit_code = 0
    except BaseException as exc:
        result.update(
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except BaseException as exc:
                exit_code = 1
                result.update(
                    status="failed", shutdown_error=f"{type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
        result["finished_at"] = time.time()
        write_json(args.output_dir / "result.json", result)
    return exit_code


def test_qwen3_npu_smoke():
    """Load the current runtime and generate a bounded offline Qwen response."""
    root = Path(os.environ["NPU_CI_SOURCE_ROOT"])
    arguments = [
        "--source-root",
        str(root),
        "--source-sha",
        os.environ["TOKENSPEED_CI_SOURCE_SHA"],
        "--install-evidence",
        str(Path(os.environ["NPU_CI_WORK"]) / "runtime/install-evidence.json"),
        "--model",
        os.environ["NPU_SMOKE_MODEL_PATH"],
        "--output-dir",
        str(Path(os.environ["NPU_CI_WORK"]) / "model"),
    ]
    for option in (
        "model-id",
        "model-revision",
        "manifest-sha256",
        "dtype",
        "sampling-backend",
        "temperature",
        "device",
        "device-count",
        "tp-size",
        "dp-size",
        "pp-size",
        "prompt",
        "max-new-tokens",
        "seed",
        "port",
    ):
        arguments.extend(
            ["--" + option, os.environ["NPU_SMOKE_" + option.upper().replace("-", "_")]]
        )
    args = parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assert (
        worker(args) == 0
    ), f"Qwen smoke failed; inspect {args.output_dir / 'result.json'}"
    result = json.loads((args.output_dir / "result.json").read_text())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as handle:
            handle.write(
                f"\n## Qwen native NPU smoke\n\nStatus: passed; source: `{args.source_sha}`.\n\n"
            )
            handle.write(
                f"Model: `{args.model_id}@{args.model_revision}`; generated tokens: {len(result['output_ids'])}.\n\n"
            )
            handle.write(
                "Smoke checks execution and bounded nonempty output. Accuracy and performance thresholds are not evaluated.\n"
            )


@pytest.mark.parametrize(
    "response, maximum, vocabulary",
    [
        ({"output_ids": [], "meta_info": {}}, 16, 100),
        ({"output_ids": [True], "meta_info": {}}, 16, 100),
        ({"output_ids": [-1], "meta_info": {}}, 16, 100),
        ({"output_ids": [100], "meta_info": {}}, 16, 100),
        ({"output_ids": [1, 2], "meta_info": {}}, 1, 100),
        (
            {
                "output_ids": [1],
                "meta_info": {
                    "completion_tokens": 1,
                    "finish_reason": {"type": "abort"},
                },
            },
            16,
            100,
        ),
        (
            {
                "output_ids": [1],
                "meta_info": {
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                },
            },
            16,
            100,
        ),
    ],
)
def test_invalid_qwen_output_is_a_failure(response, maximum, vocabulary):
    with pytest.raises(ValueError):
        validate_response(response, maximum, vocabulary)


def test_qwen_accepts_bounded_tokens_and_valid_finish():
    response = {
        "output_ids": [1, 2],
        "meta_info": {"completion_tokens": 2, "finish_reason": {"type": "stop"}},
    }
    assert validate_response(response, 16, 100) == [1, 2]
