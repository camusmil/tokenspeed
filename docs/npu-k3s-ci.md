# Ascend NPU K3s CI

The `Ascend NPU Tests` workflow at
`.github/workflows/pr-test-ascendnpu-arm.yml` uses four groups:
`scan`, `unit-test`, `model-test`, and `finish`. It replaces the host-runner
caller at the same path, avoiding a second daily NPU workflow.

`scan` runs CPU selector and adapter regressions, classifies a complete GitHub
change list, and creates the unit/model matrices. Operator, shared kernel,
runtime, engine, scheduler, dependency, and NPU CI changes select hardware;
documentation-only changes skip it. The unit worker executes operator accuracy
then performance using the existing manifest and exact expected plans. The model
worker builds its own runtime and executes native Qwen smoke then timing only
after the unit job and its required artifact upload succeed. `finish` checks all
selected job results and reports explicitly when hardware is disabled or skipped.

## Enabling a deployment

The caller accepts the official `lightseekorg/tokenspeed` repository or a fork
whose full name matches `TOKENSPEED_NPU_CI_REPOSITORY`. Hardware stays disabled
until `TOKENSPEED_NPU_K3S_CI_ENABLED=true`. The previous host-runner enable flag
alone does not enable K3s execution. `TOKENSPEED_NPU_K3S_RUNNER_LABEL` selects the
ARC scale set and defaults to `tokenspeed-pr-arm64`.

Before enabling hardware, the operator must verify the deployed ARC scale set,
trusted execution service, fixed worker image, source/dependency/model snapshots,
shared host locks, and independent reaper. Register the exact source commit,
repository and owner IDs, caller workflow ref/SHA, event, and resource limits in
the trusted service policy. A reviewed PR merge, a later main commit, and a manual
dispatch are separate identities: none inherits another version's authorization.
Keep compute admission closed until the corresponding bounded validation window.
The Run workflow button requires this file in the repository default branch.

The reusable workflow invokes the preinstalled orchestration client at
`/opt/cp11/cp12-client.js`. This is the existing deployment compatibility path;
it does not invoke a checkpoint acceptance workflow. The Runner submits only
registered `operators` and `model` profiles over HTTPS using GitHub OIDC. It has
no NPU, Kubernetes token, or host management socket. The trusted service validates
its identity and creates bounded workers; arbitrary commands or images cannot be
submitted through the workflow.

Workers receive verified read-only source, offline dependencies, fixed model
weights, a shared host lock inode, and a task-specific output directory. They
receive no GitHub, Kubernetes, or registry credentials. Each worker holds its lock
for its own task. Source copies, virtualenvs, and build caches are independent.
The service requires a durable successful operator receipt bound to the same
repository, run, exact source, and caller before accepting a model task. A
model-only rerun may reuse a successful earlier-attempt operator receipt from
that same run and source.

## Results and failure handling

A failed stage stops the remaining stages. The execution service verifies the
compressed archive, per-file hashes, source/task identity, and stage results
before issuing a durable receipt and releasing owned resources. Required-save
failure must not produce a successful receipt or discard the scene before
independent backup is verified. Required GitHub artifact upload is part of job
success. Cancellation or expired Runner leases are handled by the independent
reaper using owned task/Job/Pod UIDs. Evidence wire markers and the client output
directory retain their deployed names for compatibility.

The model profile exposes explicit quality/performance baseline interfaces.
Missing baselines remain `not_evaluated` or fail a configured formal gate. The
fixed Qwen3-0.6B example is a bounded single-device native smoke test. It does not
establish Longcat support, multi-node model execution, HCCL/RoCE, PD/L3, or long-run
quality/performance acceptance.

Test tools live in the vendor-neutral `test` extra. Existing NVIDIA/AMD installers
request `python[test]`, preserving their test-tool requirements. Native NPU
inference installs the runtime without that extra so Triton Ascend can retain its
declared pytest dependency. `pip check` remains mandatory; dependency metadata is
not overridden. Developers running the shared tests should request the `test`
extra as well.

Infrastructure lifecycle probes, fault-injection profiles, checkpoint acceptance
workflows, and temporary CI-selection validation notes are outside this daily
source tree. Deployment acceptance tools and their historical evidence can be
maintained separately. The CPU contract regressions here protect actual operator
plans, task stage boundaries, receipt validation, and model baseline behavior.
