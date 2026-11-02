# Controlled NPU CI chain

This fork validation workflow reuses the current NPU selector, operator manifest,
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
The worker retains the host lock through all four sequential stages:
operator accuracy, operator performance, model smoke, model performance timing.
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

Documentation-only updates are expected to skip the registered hardware chain.
