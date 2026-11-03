# Ascend NPU K3s CI

The daily fork workflow reuses the current NPU selector, operator manifest,
expected plans, phase executor, and reporter. The existing host entry point and
NVIDIA/AMD workflows are unchanged.

An ephemeral ARC Runner submits a registered profile over HTTPS using GitHub
OIDC. A trusted policy binds the repository and owner IDs, workflow version,
event/ref, run/attempt, and exact tested source commit. Each new source snapshot
must be independently reviewed and registered before its compute window opens.
The Runner name is an audit field, not an authorization credential.

Workers receive only a read-only source checkout, verified offline dependencies,
fixed model weights, a host lock inode, and a task-specific writable directory.
They receive no GitHub tokens, Kubernetes tokens, host sockets, or registry keys.
The daily workflow displays `scan`, `unit-test`, `model-test`, and `finish`,
matching the task grouping used by the B200 workflow. The unit task executes
operator accuracy then operator performance. Only after its required evidence
has been saved and uploaded does the model task build its own runtime and run
model smoke then model performance timing. Each worker holds its host lock for
its own task; task-local source, build caches and virtualenvs are never shared
between the two jobs. The service requires a successful operator receipt bound
to the same repository, run, source and caller workflow before admitting the
model task. A model-only rerun may reuse a successful earlier-attempt receipt
from that same run and exact source; failed or mismatched receipts are rejected.
Failures stop subsequent stages. The model stages expose quality/performance
baseline interfaces; absent baselines remain `not_evaluated`. Qwen is a bounded
single-device native smoke example, not Longcat or multi-node model acceptance.

The executor validates the complete compressed evidence archive, per-file hashes,
task/source identity, and stage results before issuing a durable receipt and
releasing owned resources. An injected required-save failure has no successful
receipt and retains its scene until an independent backup is verified. The
independent reaper handles cancellation and expired Runner leases by owned UID.

Validation PRs target the isolated `codex/cp12-k3s-20261102` branch. They do not
merge into main or switch upstream hardware CI. Public evidence excludes host
addresses, credentials, and private deployment details.

Test tools are declared in the vendor-neutral `test` extra. NVIDIA CUDA and AMD
CI installers explicitly install `python[test]`, preserving their pytest>=9 and
pytest-asyncio requirements. Native NPU inference installs the runtime package
without that extra, allowing Triton Ascend 3.2.1 to retain its declared
pytest==8.3.2 dependency. No dependency metadata is overridden, and `pip check`
remains mandatory. Developers running the shared test suite should also request
the `test` extra.


The daily entry is `.github/workflows/cp12-k3s.yml` (Ascend NPU K3s Tests).
It has no fault-injection matrix. Each stage calls the NPU-specific reusable
matrix adapter; this changes presentation without routing GPU execution through
NPU workers or giving workers management credentials. The caller and source
versions must be registered in trusted policy before hardware admission opens.
`finish` fails when a selected task fails or unexpectedly skips.

The independent Ascend NPU K3s Acceptance workflow retains the original complete
chain plus operator, model, and required-save failures. It runs only by explicit
manual dispatch or a new `npu-ci-acceptance` label event, not on every PR update.
A green injected-failure job means the expected failure and stopping behavior
were verified. The save-failure scene still requires independent evidence backup
before owned-UID cleanup. Daily and acceptance profiles have separate explicit
caller allowlists, even when they use the same source commit.

These validation PRs still target the isolated integration branch. Adding
`workflow_dispatch` does not make the Run workflow button available before the
workflow reaches the repository default branch. PR events and reruns remain the
available validation path before that merge. Register the exact source and event
again when changing branches or deploying a merged version; registration of an
old tested PR merge does not authorize a new main commit.
