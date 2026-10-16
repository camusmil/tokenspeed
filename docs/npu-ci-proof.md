# NPU runner execution proof

In `camusmil/tokenspeed`, pushes to `main` or
`codex/fork-npu-execution-log` trigger `NPU Runner SSH Proof`.
The job uses the `tokenspeed-fork-npu-setup` label and requires runner
`tokenspeed-fork-npu-01` to execute as `ci-npu`.

Before running, create `/srv/npu-ci/work/fork-slot0/ci-proof.log` as a
regular file owned by `ci-npu:ci-npu` with mode `0640`.
The job rejects missing files, symlinks, and files it cannot write.
Each successful append records UTC time, repository, host, user, runner,
run ID, attempt, event, ref, and commit SHA in both Actions output and the file.

Watch new records over SSH:

```bash
tail -n 0 -F /srv/npu-ci/work/fork-slot0/ci-proof.log
```

Verify the Actions run finishes successfully and its run ID, attempt, and SHA
match the SSH record. A record alone does not prove the final run succeeded.
Failures before the append produce no record. This workflow checks runner
execution and logging only; it does not validate CANN or NPU computation.
