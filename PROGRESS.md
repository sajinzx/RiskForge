# PROGRESS

## T0.1 Repo skeleton
Plan: package layout per §16, pyproject with v1 deps (GPU libs as extra), ruff/mypy/pytest,
pre-commit config, commit-msg hook, project memory, tracking files, one CLI smoke test.
Acceptance: `ruff check . && ruff format --check . && mypy greenqueue && pytest -q` all pass.
Notes: Python 3.11 not installed locally (3.13, 3.10 present); `requires-python >=3.11`, venv on 3.13.
`uv` not installed; pip + venv used. Lockfile deferred (see QUESTIONS.md).
Status: done (ruff, format, mypy, pytest all pass)
