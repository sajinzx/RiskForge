# GreenQueue: uncertainty-aware carbon scheduling for AI workloads

Owner: Sajin MK. Status: v1/v2 pipeline complete for Great Britain (GB); GPU measurements, other zones and
literature review are owner tasks (see "Not done").

Every number below comes from a file under `results/` (git-ignored; regenerate with
`python -m greenqueue.cli reproduce v2`). Block-bootstrap 95% CIs (7-day blocks) are in those files.
**Synthetic parts are labelled**: job energy/duration estimates are a synthetic lognormal estimator, and GPU
power is assumed from `configs/gpus.yaml` (from memory, unverified). No GPU was measured.

## Results (GB, test period 2024-01 to 2026-09)

**Carbon forecasting (E1, E2).** Persistence MAE 44.1, ridge 39.6, LightGBM (F3) 38.4 gCO2/kWh over 1-48 h.
Raw 90% quantile intervals cover only 74.6%; split conformal gives 90.7%, CQR 86.9% (under-covers,
worst 30-day window 69.7%), ACI 90.0% with worst 30-day window 85.2% at the price of wider intervals
(218 vs 160 gCO2/kWh).

**Weather (EW, 2024-03 onward, same split for both models).** Adding archived `previous_day2` weather forecasts
lowers MAE from 37.2 to 24.5, but only for horizons above 6 h; for 1-6 h it is worse (23.3 vs 17.5). F4 CQR
coverage is 85.4% (below the 90% target).

**Scheduling (E4, 60% load, oracle limited to 120 s per block).** Savings vs ASAP: oracle 11.9%, greedy 5.4%,
threshold 4.3%, simple conformal scheduler 4.3%. Violations: ASAP 6.9%, greedy 16.7%, oracle 5.5%.
Carbon-aware shifting without capacity-aware planning raises deadline violations. The oracle is a best incumbent
from a time-limited MILP, not a proven optimum. 25 blocks hit the time limit and 2 used the greedy fallback; the 120 s run gave 11.9% and 8.2%.

**Alpha tracking (E5, synthetic estimator, lower load).** The robust MPC (S4) violation rate is 0.11, 0.17, 0.39,
0.67% for alpha = 0.01, 0.05, 0.1, 0.2: monotone in alpha but far below it, i.e. conservative. Savings 7.1-9.4%
(oracle 15.6%). Under a shifted calibration the rate stays at or below 1.3%.

**Ablations (E6).** Dropping the conformal duration/energy bounds raises violations from 0.2% to 2.4% and savings
from 8.2% to 10.6%. A perfect carbon forecast gives savings 10.7% and perfect energy/duration 10.8%, so the
duration bound costs more savings than carbon forecast error does.

**Sensitivity (E7, 12h and 24h slack, load 0.4/0.6/0.8).** Savings rise with slack and fall with load; at 80%
load every scheduler violates deadlines heavily, oracle included. Full table: `results/e7/sensitivity.csv`.

## Method

`greenqueue/carbon` (data, features, F0-F4 forecasters, hand-written split conformal, CQR, ACI) ->
`greenqueue/energy` (EN0-EN2 energy models, GPU harness) -> `greenqueue/sim` (exact-accounting discrete-time
simulator, Alibaba PAI 2020 jobs with synthetic deadlines) -> `greenqueue/schedulers` (S0 ASAP, S1 greedy,
S2 threshold, S3 oracle MILP via PuLP/HiGHS, S4 robust MPC). Schedulers see only `JobView` and a `ForecastBundle`,
so no future information leaks. Forecast leakage is tested by overwriting data from the issue time onward.
Experiments are in `greenqueue/experiments`, run with `python -m greenqueue.cli run <e1..e7|ew>`.

## Limitations

- Estimator noise is synthetic; energy models were validated only on synthetic data.
- Emission factors, GPU specs and data licences are unverified (`QUESTIONS.md`, `data/MANIFEST.md`).
- NESO's own forecast (FN) has unknown vintage, so it is excluded from the main comparison.
- Single zone (GB); average, not marginal, carbon intensity.
- The MILP oracle is time-limited at higher load; `oracle_time_limited_blocks` in `summary.json` counts these.
- MPC mode was picked on the calibration year after one diagnostic looked at test weeks (disclosed in `PROGRESS.md`).
- Related work (`paper/related_work.md`) is preliminary; Li et al. 2024 already applies conformal prediction to
  carbon forecasts, so that part is not novel.

## Not done (owner)

GPU measurements on Colab (`greenqueue/energy/harness/colab_runner.ipynb`), ENTSO-E token for DE/FR/PL/ES and
E8/E9, verifying factors/specs/licences, literature review and citations, dependency lockfile (Q1).
