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

## T1.2 ENTSO-E downloader, intensity, preprocessing
Acceptance: hand-computed intensity test, pumped-storage consumption excluded, unknown fuel raises,
gap rules (2h linear, 24h seasonal, longer stays NaN), month windowing; all checks pass.
Notes: emission factors in configs/emission_factors.yaml; entries with status "assumption" (oil, waste,
other, peat, coal-derived gas, storage) are not from the IPCC table (see QUESTIONS.md Q2).
The ENTSO-E downloader is untested against the live API (no token); only mocked-query logic is tested.
Status: done

## T2.1 + T2.4 Features, metrics, conformal
Acceptance: leakage test (overwrite data from issue time on, features unchanged) passes; split conformal and
CQR hit 90%/80% within 2 pp on synthetic iid data (n=4000); pinball hand-checked.
Convention: horizon h targets label T+(h-1)h; last known label is T-1h. Seasonal-naive uses the most recent
same-hour value known at T (k=ceil(h/24) days back).
Status: done

## T2.2 + T2.3 Forecasters and tuning
Acceptance: baselines read the right columns; seasonal naive beats persistence on a periodic signal; ridge and
LightGBM beat persistence; quantiles ordered (sorted per row to prevent crossing); folds chronological and purged;
test-period guard raises; Optuna smoke run. Full 50-trial tuning is run by the E1 script on real data.
Deviation: Optuna objective is the median pinball loss (guide §8.5 gives two wordings; tuning on tau=0.5 and reusing
the structure for 0.05/0.95 was chosen for speed).
Status: done

## T4.1 + T4.2 Simulator and schedulers S0, S1, S2, simple S4
Acceptance: toy tests with hand-computed answers (single job emissions, per-step CI, queueing, deadline boundary,
pause/resume overhead exactly once, rigid-job pause and capacity violations rejected, determinism, greedy beats
ASAP on a known series, threshold pauses interruptible job). 11 tests pass.
Design: schedulers get JobView + ForecastBundle only; truth stays in Job (simulator/oracle only).
Simple S4 = greedy window with the duration upper bound.
Status: done
