# ARM Ascend NPU operator and model CI

## Workflow contract

`pr-test-ascendnpu-arm.yml` handles main pushes, main pull requests and manual
runs. One complete GitHub API change list selects operator and model tasks
separately. Each selected stage calls `run-ci-task-matrix-npu.yml` with a matrix,
an exact `checkout_ref` SHA and `timeout_minutes`. Incomplete or stale API results
fail selection. Draft PRs and disabled hardware skip runner allocation.

The reusable workflow has `max-parallel: 1` and `fail-fast: true`. The operator
suite runs all accuracy tests before all performance tests. The independent
`qwen-npu-smoke` job prepares its runtime and generates one bounded response.
When both tasks are selected, operators run first; failure or cancellation blocks
the model. A model-only change allows the operator job to be skipped.
Any required test, configuration, log or report failure stops subsequent tests.
Artifact upload and cleanup preserve the preceding failure result.

Repository variables:

- `TOKENSPEED_NPU_CI_REPOSITORY`: explicitly permits this fork; the upstream
  repository is also recognized by the entry workflow.
- `TOKENSPEED_NPU_ARM_CI_ENABLED`: must be `true`; hardware is disabled by default.
- `TOKENSPEED_NPU_ARM_RUNNER_NAME`: required exact self-hosted runner name.
- `TOKENSPEED_NPU_ARM_MODEL_CI_ENABLED`: must be `true` for automatic model jobs.
  It stays unset until model acceptance. A manual dispatch can explicitly set
  `run_model_validation: true` to validate the model before enabling automatic CI.

Both task types use `ascend-npu-ops-test`. Runner identity variables are read by
the callee. Only the caller owns the PR concurrency group; the callee cannot
cancel its caller by sharing that group.

## Task declaration and tests

The two JSON task declarations own registration, watched paths, resources,
timeouts and report/cleanup policy. `ut-tokenspeed-kernel-npu.json` selects the
operator suite; `qwen-npu-smoke.json` selects the model suite.
Concrete cases, reference calculations and sample counts live only in the two
pytest modules under `tokenspeed-kernel/test/npu/ops/`. Add an operator module and
register its path; do not add operator-name branches to the executor.

The NPU declaration intentionally remains JSON. The shared YAML discovery and
runner classification currently accept unknown non-AMD labels as NVIDIA x86.
Changing a file suffix or adding an NPU label would not isolate the task. Neither
the NVIDIA/AMD workflows nor their shared controller or conftest is modified.
Changes under existing GPU watched paths can still select GPU CI.

The registered tests expose parametrized `test_accuracy` and `test_performance`
functions. Collection does not import `torch_npu`; a generic pytest run skips
these tests when that package is absent. The dedicated executor loads its pytest
plugin without GPU parent conftest or third-party plugins, rejects zero tests and
any skip, and validates every case/configuration/sample and process exit.
Other existing NPU tests are not automatically enrolled in this suite.

```bash
python -B tokenspeed-kernel/test/npu/ops/test_scatter.py --mode all --collect-plan
python -B tokenspeed-kernel/test/npu/ops/test_scatter.py --mode accuracy
python -B tokenspeed-kernel/test/npu/ops/test_virtual_slots_to_local.py --mode performance
```

Standalone `--mode all` executes the selected module's accuracy before its
performance. The task-level entry below executes all registered accuracy first,
then all registered performance. Standalone module commands require the prepared
NPU environment and source paths; resource admission belongs to the task entry.

```bash
python -B test/ci_system/npu/pipeline.py run --work "$NPU_CI_WORK" --mode all
```

Before any calculation, the executor collects both pytest plans and freezes the
source commit, task/script/executor hashes and plan digest in the run directory.
The performance step revalidates these identities and the successful accuracy
report. Expected coverage is never inferred from completed results. The current
suite preserves 53 accuracy cases, 14 performance configurations and 70 samples.
Those numbers describe coverage, not a second manually maintained test plan.
Performance thresholds are not defined: `performance_verdict=not_evaluated` even
when every timing sample is complete and valid.

## Offline Qwen smoke

`test/runtime/models/qwen_npu_smoke.py` runs the native NPU engine with one visible
device, BF16, greedy sampling, seed 0 and TP/DP/PP equal to 1. The task fixes the
prompt and a maximum of 16 generated tokens. It validates token IDs, finish reason,
completion count, finite JSON evidence and nonempty decoded text. This is an
execution smoke check; accuracy and performance thresholds are not evaluated.

The deployment supplies the official `Qwen/Qwen3-0.6B` snapshot at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its complete content manifest has SHA-256
`5a25b58f54e9e74b4c1dce7f5ef8514bcf4e974afeaf2f5dc39e710250bd2d21`.
Model files live outside disposable run directories, owned by root and readonly.
Every model job verifies the fixed manifest and every listed file, then loads
weights and tokenizer offline with remote code disabled.

`Prepare model runtime` creates a fresh job virtualenv. It installs the reviewed
NPU wheels and the current projects' declared build requirements, then builds
TokenSpeed, scheduler and both kernel packages from the exact checkout. Standard
pip resolves all runtime dependencies and `pip check` must pass. Build isolation
is disabled after installing the declared build requirements because the kernel
build invokes pip itself. Dependency resolution remains enabled. The scheduler
uses the existing OpenSSL development directory from host policy.

The model environment loads CANN native libraries and IR, while excluding CANN
profiler Python distributions from its Python search path. This gives inference
its own dependency resolution. The runtime pytest minimum is 8.3.2 to permit the
reviewed Triton-Ascend version; pytest 9 remains permitted in other environments.
The operator environment remains independently prepared.

Install evidence verifies local source origins, the checkout SHA, kernel package
SHA and scheduler binary hash/library loading. The model script checks this
provenance again and verifies actual native NPU arithmetic before generation.
All writable build files, virtualenv and caches belong to the current task.

## Host admission and source preparation

The prepared deployment is Linux ARM64, account `ci-npu`, environment
`/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`, and its Python virtualenv.
Each phase verifies a clean exact checkout, loads CANN/Python, then acquires the
shared `/srv/npu-ci/locks/host-all.lock` with a bounded wait. Matrix concurrency
is not a replacement for this cross-runner lock and does not promise FIFO.
The phase execution budget starts after admission. Current task budgets are
600 seconds per lock wait, 1800 seconds per operator phase and 90 minutes per
job. The model has a separate 900-second installation deadline and 600-second
inference deadline. Lock waiting does not consume the inference deadline.

Before sources exist, the workflow uses a root-owned installation of the same
`pipeline.py` under `/srv/npu-ci/runtime/vm-phase1-qwen-v1`. Its `policy.json` contains
deployment paths, repository identity, minimum free space, `task_paths`, the
readonly model mapping and model build paths. `task_paths` maps each supported
config to its root-owned declaration copy; the legacy `task_path` may still
identify the operator declaration. Archive policies must agree across tasks. Installation is an operator action,
not a root privilege granted to repository code. Runner user execution cannot
select arbitrary host deletion roots.

The trusted `prepare` command validates the matrix identity and creates an owned
run directory before fetching. Task names are part of work, evidence and receipt
identities, so two tasks in one Actions run cannot overwrite each other.
Git fetch makes at most four attempts, separated by 30 seconds, retaining each
Git error and exit code. Cancellation is not retried. Each download attempt is
bounded, and a final failure retains bootstrap/fetch diagnostics for archiving.
It checks the exact checkout SHA and byte equality of the installed pipeline and
task declaration with the selected source. A source change to either requires
an explicit reviewed runtime update before hardware execution. The Qwen version
uses its own runtime and work/evidence/state slot; older validated slots remain
available to their existing workflows.

Each test subprocess runs in a private process group. A separate supervisor
holds the shared lock and watches a parent pipe and the owning Runner.Worker
PID/start time/boot identity. Controller death, including SIGKILL, triggers group
reclamation. This is independent of Artifact availability. No runtime setup
restarts the Runner, changes CANN/drivers, or rewrites older deployment slots.

## Evidence and cleanup

The reusable workflow performs official `actions/upload-artifact` uploads. Policy
comes from the task: 7 days retention, 1 GiB maximum bundle size and three upload
attempts. The three upload steps temporarily tolerate an attempt failure only
for retry; final acknowledgement returns failure if every upload failed.

Successful runs upload summaries, context and run identity. Model runs also
retain install provenance, model manifest, admission and generation results. Failed runs include
raw diagnostics, detailed reports and the frozen plan when available. Logs use
one physical local copy; bundle selection hardlinks files on the same filesystem.
Active producers and symlinks are rejected. Manifests and hashes are checked
before cleanup. Artifact ID/digest and the minimal receipt survive outside the
disposable directories.

Only after successful upload acknowledgement are this run's checkout, cache,
temporary files, logs and bundle removed. Upload failure retains the only local
evidence. The separate report-only timer warns after 24 hours and waits for
manual handling; it never deletes unarchived evidence. Historical directories,
prepared environments, readonly model files and existing runners are preserved.
No cross-run source or compilation cache is implemented.

Infrastructure regression scripts are maintained outside the formal source
patch. Nightly, K3s and branch-protection work remain separate.
Acceptance must name the new commit and distinguish GitHub logs/artifact
from the independently read VM receipt; cleaned VM raw logs cannot be claimed as
still available for a line-by-line comparison.
