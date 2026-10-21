# ARM Ascend NPU operator CI

## Workflow contract

`pr-test-ascendnpu-arm.yml` handles main pushes, main pull requests and manual
runs. It uses the complete GitHub API change list to select the suite and emits
an `include` matrix, an exact `checkout_ref` SHA and `timeout_minutes`. Unrelated
changes, draft PRs and disabled hardware skip the reusable call before requesting
a hardware runner. Incomplete or stale API results fail selection.

`run-ci-task-matrix-npu.yml` accepts those three required `workflow_call` inputs.
It expands the matrix once, with `max-parallel: 1` and `fail-fast: true`. The current
matrix contains one `ut-tokenspeed-kernel-npu` suite on `ascend-npu-ops-test`, not
one item per operator or phase. Runner identity variables are explicitly read
from repository variables in the callee. Only the caller has PR concurrency;
the callee does not reuse that group or cancel its own caller.

The same job prepares exact sources, runs **all accuracy** tests, then **all
performance** tests. Any required test, configuration, log or report failure
stops subsequent tests. Performance is skipped after accuracy failure. Official
Artifact uploads and final cleanup cannot override a failed test result.

Repository variables:

- `TOKENSPEED_NPU_CI_REPOSITORY`: explicitly permits this fork; the upstream
  repository is also recognized by the entry workflow.
- `TOKENSPEED_NPU_ARM_CI_ENABLED`: must be `true`; hardware is disabled by default.
- `TOKENSPEED_NPU_ARM_RUNNER_NAME`: required exact self-hosted runner name.

Draft PR hardware remains disabled. A manual run on the selected branch is used
for hardware acceptance without converting the PR to ready.

## Task declaration and tests

`test/ci/npu_test/ut-tokenspeed-kernel-npu.json` is the single source for task
registration, watched paths, resources, timeout and report/cleanup policy.
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

## Host admission and source preparation

The prepared deployment is Linux ARM64, account `ci-npu`, environment
`/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`, and its Python virtualenv.
Each phase verifies a clean exact checkout, loads CANN/Python, then acquires the
shared `/srv/npu-ci/locks/host-all.lock` with a bounded wait. Matrix concurrency
is not a replacement for this cross-runner lock and does not promise FIFO.
The phase execution budget starts after admission. Current task budgets are
600 seconds per lock wait, 1800 seconds per phase and 90 minutes per job,
leaving 10 minutes for preparation and archiving.

Before sources exist, the workflow uses a root-owned installation of the same
`pipeline.py` under `/srv/npu-ci/runtime/vm-phase1-v06`. Its `policy.json` contains
only deployment paths, repository identity, minimum free space and `task_path`.
That path points to a root-owned copy of the task declaration; artifact policy
is not duplicated in a separate source file. Installation is an operator action,
not a root privilege granted to repository code. Runner user execution cannot
select arbitrary host deletion roots.

The trusted `prepare` command creates an owned run directory before fetching.
Git fetch makes at most four attempts, separated by 30 seconds, retaining each
Git error and exit code. Cancellation is not retried. Each download attempt is
bounded, and a final failure retains bootstrap/fetch diagnostics for archiving.
It checks the exact checkout SHA and byte equality of the installed pipeline and
task declaration with the selected source. A source change to either requires
an explicit reviewed runtime update before hardware execution.

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

Successful runs upload summaries, context and run identity. Failed runs include
raw diagnostics, detailed reports and the frozen plan when available. Logs use
one physical local copy; bundle selection hardlinks files on the same filesystem.
Active producers and symlinks are rejected. Manifests and hashes are checked
before cleanup. Artifact ID/digest and the minimal receipt survive outside the
disposable directories.

Only after successful upload acknowledgement are this run's checkout, cache,
temporary files, logs and bundle removed. Upload failure retains the only local
evidence. The separate report-only timer warns after 24 hours and waits for
manual handling; it never deletes unarchived evidence. Historical directories,
prepared environments and existing runners are preserved. No cross-run source or
compilation cache is implemented.

Infrastructure regression scripts are maintained outside the formal source
patch. Model, nightly, K3s and branch-protection work are outside this operator
subset. Acceptance must name the new commit and distinguish GitHub logs/artifact
from the independently read VM receipt; cleaned VM raw logs cannot be claimed as
still available for a line-by-line comparison.
