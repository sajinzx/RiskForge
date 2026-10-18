# GreenQueue: project memory

Single source of truth: `implementation.md`. Read it fully before writing code.
Owner: Sajin MK (only Git contributor).

## Commands
- Setup: `python -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"`
- Check: `ruff check . && ruff format --check . && mypy greenqueue && pytest -q`
- CLI: `greenqueue run eNN`, `greenqueue reproduce v1`

## Loop (implementation.md §13)
Per task: read, plan (PROGRESS.md), test, build, check, verify, decide.
Stop and write to QUESTIONS.md at any decision gate (§3.4) or after 3 failed attempts.

## Hard rules
- Never invent data, results, emission factors or citations.
- The scheduler only sees forecasts (`ForecastBundle`); only the simulator sees truth.
- Never commit `.env`, `data/raw/`, `data/processed/`, large results.
- Never change `git config user.*`; never rewrite history or force-push without explicit instruction.
- Commits: one per task step, Conventional Commits (`type(scope): summary`, body, `Refs: Tx.y`).
  Messages describe only what was done. No co-author trailers and no AI attribution of any kind
  (the `commit-msg` hook rejects them). Enable the hook with `git config core.hooksPath scripts/hooks`.
- Branch per task (`t2.3-lightgbm-quantile`); `main` stays runnable.
