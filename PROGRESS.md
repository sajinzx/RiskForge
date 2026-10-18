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

## T4.3 Oracle MILP (S3)
Acceptance: known optima (rigid job, interruptible split, capacity sharing); randomized check that no online scheduler
beats the oracle when both meet all deadlines (4 seeds); 50 tests pass.
Notes: PuLP pinned to 3.x (4.0.1 changed the API). Oracle blocks of 168 arrival steps with reserved capacity carried
forward. With checkpoint overhead > 0 the plan is near-optimal (partial last step charged as full), documented.
Status: done

## T5.1 + T5.2 Trace loader and job synthesis
Real data: Alibaba PAI cluster-trace-gpu-v2020 `pai_task_table` (34 MB) downloaded to data/raw/traces/alibaba;
SHA-256 matches the value published in the trace README (cd1d6dc3...).
Choices (owner to review): each task is one job; terminated tasks only, 10 min to 72 h; GPUs = ceil(inst_num*plan_gpu/100);
trace weeks resampled with replacement onto target weeks (weekday and time of day kept, trace clock UTC+8 treated as
the target zone's clock); 150 jobs per week subsampled so the oracle MILP stays tractable; pool sized for 60% ASAP utilisation.
Power per GPU is an ASSUMED lognormal (median 0.20 kW) because no GPU measurements exist yet.
Status: done

## T5.3 E1 and E4 pipelines (code + smoke tests)
E1: GB hourly series (quality report: 0.5% missing before cleaning, 243 h still missing), F0-F3 + conformal,
7-day block-bootstrap CIs, coverage/calibration/MAE plots, forecast export for scheduling. E4: trace replay with
S0-S4, paired block-bootstrap savings, oracle share. Both have end-to-end tests on synthetic inputs.
Finding: raw LightGBM 5/95 quantiles under-cover (74% at nominal 90% in the smoke run); split conformal and CQR
restore about 92-93%.
Finding: NESO archived forecast has unknown vintage (flat error 12-48 h) and is excluded from main results (Q3).
Status: code done; full runs recorded below when finished.

## T3.1 Measurement harness
Built: NVML power logger + trapezoid integration of (power - idle) with warm-up discard, resumable sweep (JSONL, skips
finished config/repeat pairs), sweep grid from guide 9.1 (8 + 3 + 9 + 32 configs), torch workload runners (random data,
FLOPs via torch.utils.flop_counter), Colab notebook, GPU spec config.
Tested without a GPU: integration hand-computed, logger rate, grid size, resumability, output fields.
NOT tested on a GPU: workloads.py (torch runners) and NvmlSampler; they need the owner's Colab run (T3.2).
Deviation: idle power is measured once per config (30 s), not before every repeat, to keep the sweep near 3 hours.
GPU peak-FLOP and TDP values in configs/gpus.yaml are from memory: owner must verify.
Status: code done; measurements pending (owner).

## T3.3 Energy models, validation, conformal (+ T8.2 gray-box early)
Built: measurement aggregation (per-config means/stds, MAD outlier flag), EN0 log-linear, EN1 LightGBM, EN2 gray-box,
random / LOMO / LOGO folds, group-conditional conformal on log-residuals (family and GPU, pooled fallback below 20 points),
E3 experiment. Gray-box extensions beyond the guide (documented in the class): a baseline-power parameter p0 and
shrunk per-family efficiency multipliers. Gray-box is compute-bound only: real LLM decoding is memory-bound and will
likely fit poorly until a memory term is added (needs bytes/step in the harness).
IMPORTANT: no real energy data exists yet. Every energy result so far is SYNTHETIC (generator follows compute-bound physics,
so accuracy there is circular and proves nothing). `greenqueue run e3` fails with a clear error until measurements exist;
`greenqueue run e3 --smoke` writes to results/e3_SYNTHETIC. On synthetic data conformal coverage drops under LOMO/LOGO
(about 0.65-0.70 at nominal 0.90), the expected shift effect.
Status: code done, real measurements pending (owner, T3.2).

## T5.4 Fix: cap per-job GPU requests (found by reading the first E4 run)
First full E4 run showed pool sizes of 100-400 GPUs and ASAP utilisation of 2-13% instead of 60%: a few huge distributed
Alibaba tasks set the pool size, so capacity never bound and the comparison was too easy. Per-job GPU requests are now capped
at 8 (configs/sim.yaml max_job_gpus; capped jobs counted in meta). The uncapped run is kept in results/e4_superseded_uncapped
for reference and must not be quoted.
Status: fixed; E4 rerun below.

## T7.4 ACI with delayed feedback + E2
ACI per horizon bucket on CQR scores; an outcome is usable only once its target hour ended (test proves intervals do not
change when unobserved outcomes are corrupted); synthetic shift test: static coverage < 0.75, ACI > 0.86. E2 picks gamma on
the calibration year (scores from first half, ACI over second half) and compares U1/U2/U3 on the test period.
Status: code and tests done; real run recorded below.

## T9.2 E5 pipeline + parallel sweep runner
sweeps.py runs (cell, seed) tasks in parallel (workers reload inputs from disk). E5 sweeps alpha in {0.01, 0.05, 0.1, 0.2}
for simple S4 and the MPC S4, in a `calibrated` and a `shifted` (errors 1.6x larger than calibration) scenario, against
S1 (no bound), S0 and the oracle. The duration estimator is a SYNTHETIC lognormal-noise stand-in (no energy model yet).
Real E2 result (test 2024-01..2026-09, nominal 90%): coverage static CQR 0.869, split conformal 0.907, ACI 0.900;
worst 30-day coverage 0.697 / 0.823 / 0.852; ACI intervals ~35% wider than CQR. gamma=0.01 chosen on the calibration year.
Status: code and smoke tests done; full run recorded below.

## T10.1 E6 ablations and E7 sensitivity (code)
E6 variants of the robust MPC at 30% load, alpha 0.05: full, no conformal bounds, carbon point only, carbon CVaR (normal
approximation from the 90% interval), perfect energy/duration, static CQR carbon, perfect carbon forecast. No-weather is
not run (F4 weather model not built). E7: slack x load grid for S1, simple S4 and oracle plus interruptibility 0/1; zone
axis not run (GB only). Both use the SYNTHETIC duration-noise estimator. Smoke tests pass.
Status: code done; full runs recorded below.

## T4.3b Oracle: lexicographic lateness, E4 on the parallel runner
Probe on 150-job weekly blocks at 60% load: blocks took 10-120 s; some hit the time limit and are now flagged
(`oracle_time_limited_blocks`). Single-phase penalty weights of 1e7 per late step made the solver's 1% gap meaningless for
carbon; replaced by: feasibility with zero lateness, else minimum lateness, then minimum carbon with lateness fixed.
E4 now runs seeds in parallel through sweeps.run_cells and records oracle-quality counters.
Status: done; full E4 rerun below.

## T9.3 MPC carbon risk level chosen on the calibration year
Disclosure: the first E5 run used the guide default (plan with the 90% upper carbon bound). It saved only 1.6-2.2% against
10.6% for the greedy window. A diagnostic on test-period weeks 0-2 showed this came from planning with the upper bound
(its width grows with horizon, so far hours look dirty and shifting is discouraged): with perfect forecasts the MPC saved
12.4% vs 11.2% for greedy with 0% violations, so the implementation is sound. To avoid selecting the setting on test data,
the carbon mode (point / cvar / upper) is now chosen on the 2023 calibration year (static CQR forecasts, seeds 100-101) with
the rule "highest saving among modes at most 1 pp above ASAP's violation rate", and E5/E6 use that choice. The first E5
output is superseded and must not be quoted.
Status: done; E5/E6/E7 full runs follow.

## Fix: E5 load override lost (found by comparing E5 with E6)
After reformatting, the scenario cells of E5 lost `overrides=(UTILISATION,)`, so the robust schedulers ran at 60% load while
the ASAP baseline ran at 30%: savings and violations in that E5 output are invalid (kept as results/e5_superseded_load_bug).
Both E5 and E6 now assert that all cells of an experiment see the same GPU pool per seed. E5 is rerun after E7 and the
long-oracle E4. E6 and E7 were unaffected (their overrides were present and are covered by the new assertion for E6).

## Oracle fallback for blocks with no MILP incumbent
At 60-80% load some oracle blocks return no solution inside the time limit (HiGHS "Not Solved"), which crashed E7 and the
long-limit E4. Such blocks now use a deterministic greedy plan (cheapest feasible windows/steps in deadline order),
counted as `oracle_heuristic_blocks` and treated like time-limited blocks: the oracle is then an incumbent, not a bound.
E4 reports both counters; the "no scheduler beats the oracle" check is waived when any block was not proven optimal.
