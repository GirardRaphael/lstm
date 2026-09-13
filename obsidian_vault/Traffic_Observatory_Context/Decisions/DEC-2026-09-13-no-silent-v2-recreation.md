---
id: dec-2026-09-13-no-silent-v2
type: decision
status: accepted
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Decision: Do not silently recreate pipeline v2

## Problem

Continuity notes describe a substantial unpublished `codex/temporal-pipeline-v2` branch that is not present locally or on GitHub.

## Evidence

Branch/commit search failed; no `test_temporal.py`; `TrainingConfig` on `a652a04` has no `pipeline_version`.

## Alternatives

1. Recreate v2 from handoff prose alone
2. Wait for user-supplied bundle/checkout
3. Implement a new causal pipeline under a fresh version id without claiming identity with `646d3a7`

## Choice

**(2)** now; **(3)** only if the user explicitly authorizes reconstruction and accepts that it is not bit-for-bit continuity with the missing branch. Reject **(1)**.

## Consequences

M1 is blocked on recovery or an explicit rebuild decision. M0 documentation and baseline verification can still complete.

## Rollback / revisit

Revisit immediately when a bundle, patch, or path to the unpublished worktree is provided.
