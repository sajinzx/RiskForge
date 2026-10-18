# PROGRESS

## T0.1 Repo skeleton
Plan: package layout per §16, pyproject with v1 deps (GPU libs as extra), ruff/mypy/pytest,
pre-commit config, commit-msg hook, project memory, tracking files, one CLI smoke test.
Acceptance: `ruff check . && ruff format --check . && mypy greenqueue && pytest -q` all pass.
Notes: Python 3.11 not installed locally (3.13, 3.10 present); `requires-python >=3.11`, venv on 3.13.
`uv` not installed; pip + venv used. Lockfile deferred (see QUESTIONS.md).
Status: done (ruff, format, mypy, pytest all pass)

## T1.1 UK API downloader
Plan: `greenqueue/carbon/uk_api.py` (windowing, retry/backoff, parsing, stitching, row-count report)
plus thin script `data/scripts/download_uk_ci.py`; mocked-HTTP tests.
Acceptance: tests cover windowing, leading-slot trim, dedupe, NaN actuals, retry/no-retry, stitching;
ruff/mypy/pytest pass; live 2-day smoke run gives 96/96 rows.
Notes: the API returns the slot containing `from` (starts before `from`), so rows are trimmed to
`[from, to)`. Full 2018 to 2026-09 download is for the owner (T1.3); not run here.
Status: done
