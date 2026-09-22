# Path-based area labeling

Cirq applies `area/*` and `interface/*` labels automatically from repository paths.

## Configuration

Rules live in [`.github/labeler.yml`](../../.github/labeler.yml), separate from the GitHub Actions workflows. Each label maps to one or more path globs in the format expected by [`actions/labeler` v5](https://github.com/actions/labeler).

When adding or changing rules:

1. Use labels that already exist on the repository.
2. Prefer specific sub-areas (for example `area/google/engine`) in addition to broader parent labels (for example `area/google`) when both apply.
3. Keep globs aligned with [`.github/CODEOWNERS`](../../.github/CODEOWNERS) where practical.

## Pull requests

The [Pull request labeler](../../.github/workflows/pr-labeler.yaml) workflow runs `actions/labeler` on opened and updated pull requests. Matching labels are added on each update. Label removal (`sync-labels`) is intentionally disabled for now so manually added `area/*` labels are never stripped automatically; we can enable syncing later if maintainers want area labels to track the current diff exactly (the size labeler already syncs its own labels this way).

## Issues

The [Issue labeler](../../.github/workflows/issue-labeler.yaml) workflow scans the issue title and body for repository path references and applies the same rules via [`dev_tools/path_area_labeler.py`](../path_area_labeler.py). Mention paths such as `cirq-core/cirq/ops/common_gates.py` or `` `docs/dev/triage.md` `` when filing bugs so the right area labels are applied.

Issue labeling is additive on open: labels are not removed automatically if the issue text changes later.

## Local testing

```bash
python dev_tools/path_area_labeler.py --dry-run --text "Bug in cirq-core/cirq/sim/sparse_simulator.py"
pytest dev_tools/path_area_labeler_test.py dev_tools/labeler_config_test.py
```
