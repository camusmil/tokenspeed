# Ascend NPU operator CI

The ARM workflow runs the registered operators on one prepared Linux VM. It
executes **all accuracy tests before any performance tests**, with global
fail-fast behavior. A failed test, incomplete result, timeout, or required
evidence write stops subsequent tests and fails the job. Performance currently
validates complete, finite timing samples; `performance_verdict` remains
`not_evaluated`. There is no performance acceptance threshold.

## Entry points and selection

`.github/workflows/pr-test-ascendnpu-arm.yml` handles pushes to `main`, pull
requests targeting `main`, and manual dispatch. The selection job obtains changed
paths through the GitHub API, including rename sources. Missing or truncated API
results and stale pull request identities fail selection. Unrelated changes skip
hardware explicitly. Manual dispatch selects the operator suite.

Hardware requires `TOKENSPEED_NPU_ARM_CI_ENABLED=true`, the
`ascend-npu-ops-test` runner label, and the exact runner name configured in
`TOKENSPEED_NPU_ARM_RUNNER_NAME`. Draft pull requests skip hardware; this is not a
hardware pass. A manual dispatch of the same branch can validate hardware without
changing draft status. A fork must set `TOKENSPEED_NPU_CI_REPOSITORY` to its own
`owner/repository`. The workflow does not change GPU workflows or shared CI
controllers. Files under `test/**` can still trigger existing GPU path rules.

## Files and task contract

- `test/ci/npu_test/ut-tokenspeed-kernel-npu.json` is the single task declaration:
  operator registration, watched paths, stages, VM resources, and timeouts.
- `test/ci/npu_test/test_case/test_*.py` contains actual operator definitions,
  reference calculations, and timing parameters.
- `test/ci_system/npu/ci_path_filter.py` selects tasks from complete API results.
- `task_config.py` validates the declaration. `run_ci_task.sh` checks the VM and
  acquires the shared host lock. `pipeline.py` executes and validates each stage.
- `operator_cli.py` and `ci_test_plan.py` collect a plan without importing device
  libraries or performing NPU computation. `ci_reporting.py` verifies results.
- `ci_lifecycle.py` manages process supervision, artifact selection, upload
  receipts, and cleanup. `artifact-policy.json` records the archive policy.

Register another operator by adding its script and one task entry. Each script
must implement explicit `--mode accuracy`, `--mode performance`, and `--mode all`,
plus `--mode all --collect-plan`. There are no operator-name branches in the
executor. Infrastructure acceptance scripts are maintained outside the production
patch; the workflow has no dependency on those scripts.

Before accuracy execution, both stage plans are collected from the selected
scripts and saved in `frozen-plan.json` outside the checkout. The plan binds the
source commit, task hash, script/executor hashes, named cases/configurations,
metadata, and timing requirements. Empty or duplicate plans fail. Performance
requires a successful accuracy report and the same frozen-plan digest. Source or
plan changes during execution fail validation. Actual records must match the
planned order and parameters; missing, duplicate, extra, failed, or non-finite
samples are rejected. The plan does not prevent an author from intentionally
changing test coverage in a source change; review remains responsible for that.

The initial task uses device 0 while holding the shared whole-host lock. The
registered scatter and virtual-slots tests retain their existing cases and timing
parameters. Model evaluation and nightly stability workloads are outside this
operator-only implementation.

## Prepared VM contract

The runner must use the `ci-npu` account on Linux ARM64. The existing environment
entry is `/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`, and its interpreter
is the adjacent `venv/bin/python`. The checkout's kernel source directories are
placed first on `PYTHONPATH`. Reports record the source/task/plan identities,
Python and package versions, environment-entry hash, and device allocation.

All cooperating runners must use `/srv/npu-ci/locks/host-all.lock`. Each test stage
waits at most 600 seconds for this lock and then has a separate 1800-second
execution budget. Waiting is reported independently. The GitHub job timeout is
90 minutes, including source preparation and finalization. File locking does not
promise FIFO ordering or implement priority/preemption.

The bootstrap fetches the exact `GITHUB_SHA` into a new checkout. Fetch failures
allow three retries after the initial attempt, with 30 seconds between attempts;
each original Git error and exit code is retained. Cancellation is not retried.
There is no shared source cache, shared compilation cache, or reuse of a mutable
previous checkout. The prepared CANN environment and runner installation remain
outside per-run cleanup.

Linux test subprocesses run under a separate supervisor. A pipe detects phase
controller death, including `SIGKILL`; the supervisor also checks the owning
Runner.Worker's PID, start time, and boot identity. It terminates the test process
group before releasing its inherited host lock. Normal completion, cancellation,
and timeout also clean up that group. This protects against controller/Worker
exit; simultaneous termination of every supervisor or a host crash is outside
that live-process guarantee. Unarchived files remain discoverable by the pending
check and require operator handling; they are never inferred to be archived.

## Artifact policy and finalization

There are two top-level test steps and one archive finalization step. Before
source download, the trusted host lifecycle runtime generates a temporary local
composite action in the runner workspace. This action uses the pinned official
`actions/upload-artifact` implementation. No generated action or acceptance helper
is committed as a separate source directory.

Successful runs upload phase summaries and minimal context/identity information.
Failed or interrupted runs upload available detailed logs, reports, and the
frozen plan. Source-fetch failures include bootstrap diagnostics. Raw logs are
hard-linked into the upload bundle on the same filesystem, so there is one
physical log copy. Collection refuses to proceed while a registered producer is
still active, and file digests are checked again before cleanup.

Artifacts are retained for 7 days, with a 1 GiB total selected-file limit. Upload
has at most three step-level attempts; each attempt uses a distinct artifact
name. The underlying official action may also retry its own transport requests.
Only a successful upload with an artifact ID and digest permits removal of this
run's checkout, temporary files, cache, logs, and local bundle. A minimal receipt
is retained separately. Cleanup failure is reported as a finalization failure.

If collection or upload fails, the job fails and local evidence remains. After
24 hours, the pending check raises an operator-action warning; it does **not**
automatically delete unarchived evidence. The 24-hour value is a warning boundary,
not permission to destroy the only log copy. Upload failure does not hold the NPU
lock after computation has stopped.

## Host runtime installation and troubleshooting

Install the reviewed `ci_lifecycle.py` and `artifact-policy.json` as root-owned,
non-writable-by-runner files under `/srv/npu-ci/runtime/vm-phase1`. Create a
root-owned `policy.json` that combines the archive policy with the actual
`repository`, `jobs_root`, `evidence_root`, `state_root`, and
`minimum_free_bytes`. Jobs/evidence/state directories must be separate from older
runtime directories and writable by `ci-npu`; jobs and evidence must share a
filesystem. The bootstrap verifies that the selected commit's lifecycle source
and artifact policy exactly match the installed runtime before testing.

The host's periodic pending check invokes the installed helper with
`--policy /srv/npu-ci/runtime/vm-phase1/policy.json gc --mode apply`. It reports
unarchived runs, preserves active or unknown owners, and does not touch runner
diagnostic logs, older runtimes, historical evidence, or other jobs. A deployment
update must preserve those boundaries and must not restart the runners.

For failures, inspect the failed test step and its summary, then the run's
`npu-evidence-<run>-<attempt>-upload-<n>` artifact. An absent artifact with a failed
finalizer requires checking the managed work directory on the VM. A deployment
identity mismatch requires updating the reviewed host runtime; a source-fetch
failure is not an operator failure, and retry success is not proof of a permanent
network repair.
