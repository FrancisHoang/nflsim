# Contributing

## Workflow

- One feature per branch, named `feat/<short-name>` (or `fix/`, `docs/`, ...).
- Keep pull requests small and independently mergeable.
- Write tests alongside the code using small in-memory fixtures. **Tests must never use
  the network**: an autouse fixture in `tests/conftest.py` blocks it.
- Put tunable numbers in the config file, not in code.
- Every simulation takes an explicit RNG seed.
- Never use information from after kickoff when simulating a game (no future-data leakage).
- Any change to the model must be backtested against the previous phase, and the result
  reported honestly in the PR, including when it doesn't help.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(data): add parquet cache for nflverse loaders
fix(cache): refetch stale current-season files
docs: describe injury data gaps
test(cli): cover --refresh flag
```

## Before pushing

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

## Merging

- `main` is protected: changes land only through pull requests, and both CI checks
  (Python 3.11 and 3.12) must pass with the branch up to date with `main`.
- Pull requests are **rebase-merged**, so `main` keeps a linear history of the
  individual conventional commits. Branches are deleted automatically after merging.
- For stacked pull requests, retarget the child to `main` before merging the parent,
  then rebase the child onto `main`.

## Releases

Each finished roadmap phase is a minor release (`v0.N.0`): bump the version in
`pyproject.toml` and `src/nflsim/__init__.py`, run `uv lock`, add a `CHANGELOG.md`
entry with the phase's backtest results, then tag the merged commit and publish a
GitHub release.
