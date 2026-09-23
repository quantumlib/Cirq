# Path-based area labeling

Cirq applies `area/*` and `interface/*` labels automatically from repository paths on
pull requests.

## Configuration

Rules live in [`.github/labeler.yaml`](../../.github/labeler.yaml), separate from the GitHub
Actions workflows. Each label maps to one or more path globs in the format expected by
[`actions/labeler`](https://github.com/actions/labeler).

When adding or changing rules:

1. Use labels that already exist on the repository.
2. Prefer specific sub-areas (for example `area/google/engine`) in addition to broader parent
   labels (for example `area/google`) when both apply.
3. Keep globs aligned with [`.github/CODEOWNERS`](../../.github/CODEOWNERS) where practical.

## Pull requests

The [Pull request labeler](../../.github/workflows/pr-labeler.yaml) workflow runs
`actions/labeler` on opened and updated pull requests. Matching labels are added on each
update. Label removal (`sync-labels`) is intentionally disabled for now so manually added
`area/*` labels are never stripped automatically; we can enable syncing later if maintainers
want area labels to track the current diff exactly (the size labeler already syncs its own
labels this way).

Wide refactors that would add more than five changed-files labels skip area labeling for
that run; see `changed-files-labels-limit` in `.github/labeler.yaml`.
