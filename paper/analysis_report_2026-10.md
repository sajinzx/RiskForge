# GreenQueue analysis report: full rerun, more GPUs, Philly, ENTSO-E (2026-10-20 to 2026-10-25)

Prepared by Sujit Narrayan M (collaborator). Follows `paper/status_report.md` (2026-10-19).
All experiments E1 to E9 were rerun from raw data on one Windows machine (8 cores, 16 GB RAM).
`results/`, `data/raw/` and `data/processed/` stay out of git by the repository rules, so this file
records the numbers. Every number below is read from `results/<experiment>/` of that run.

## What is new since the status report

| Item | Roadmap task | State |
|---|---|---|
| ENTSO-E token received; DE_LU, FR, PL, ES downloaded 2018-01 to 2026-09 | T0.4, T1.3, T7.2 | done |
| E9 cross-zone transfer run on 5 zones | E9, T10.1 | done |
| Full T4 sweep on Kaggle (156 runs, 52 configs, 3 repeats) | T3.2 | done |
| Three more GPU types measured on laptops (RTX 3050, 4050, 5070 Laptop) | T8.1 | partial (see below) |
| LOGO (leave-one-GPU-out) validation now defined | T3.3, T8.2 | done |
| Philly trace loader and E4 on Philly | T5.1 | done |
| E1, E2, E4, E5, E6, E7, E8 rerun on freshly downloaded GB data | T5.3, T9.2, T10.1, T11.1 | done |

## Data

**Carbon intensity, hourly mean 2018-2026 (gCO2/kWh):** FR 49, ES 169, DE_LU 325, PL 623, GB 203 (train
mean). Gaps after cleaning: FR 25 h, ES 35 h, DE_LU and PL none. DE_LU has no data before 2018-09 (the
bidding zone starts then). GB: 0.12% of half-hours missing before cleaning.

**Job traces:** Alibaba PAI v2020 task table (SHA-256 matches `data/MANIFEST.md`), 368,238 tasks over 11
weeks after filtering. Philly (`msr-fiddle/philly-traces`): 49,007 finished jobs over 16 weeks (13 weeks
have enough jobs for the 150-jobs-per-week sampling).

**GPU energy measurements (`data/raw/energy_own/`):**

| GPU | Runs used | Notes |
|---|---|---|
| Tesla T4 (Kaggle) | 156 | full grid, 3 repeats; repeat spread 0.4% median |
| RTX 4050 Laptop | 81 of 90 | owner of the laptop opened other programs during some runs; 9 repeats dropped by the outlier rule |
| RTX 3050 Laptop | 26 of 30 (day 1) | the laptop shut down mid-sweep; after restart NVML reported a 30 W power limit instead of 70 W and repeats differed by ~45%, so the 24 day-2 runs (all LLM) are kept in `data/raw/energy_excluded/` and not used |
| RTX 5070 Laptop | 11 of 12 | BERT only: the TinyLlama download on that laptop was corrupt, so the sweep stopped early |

## Results

### E1 carbon forecasting (GB, test 2024-01 to 2026-09)
F3 LightGBM MAE 38.4 (37.2-39.7) vs persistence 44.1. Interval coverage at nominal 90%: raw 0.746,
split conformal 0.907, CQR 0.869. All sanity checks pass.

### E2 coverage under shift
Split 0.907, CQR 0.869, ACI 0.900. Worst 30-day coverage: 0.823 / 0.696 / 0.852. ACI intervals are about
35% wider than CQR.

### E3 energy model generalisation (4 GPU types, 84 configurations)

| Model | Random medAPE | LOMO medAPE | LOGO medAPE | LOGO coverage (nominal 0.90) |
|---|---|---|---|---|
| EN0 log-linear | 60% | 80% | 67% | 0.99 |
| EN1 LightGBM | 48% | 85% | 67% | 0.89 |
| EN2 gray-box | 60% | 90% | 69% | 0.98 |

Intervals keep coverage on unseen GPUs but point errors are 50-70%. LOGO interval width is infinite for
some folds (small groups). EN2 depends on the laptop GPU peak-TFLOPS and TDP values, which are
assumptions (see "Needs the owner").

### E4 scheduling on real traces (13 test weeks, 5 seeds, GB carbon)

| Scheduler | Alibaba saving | Alibaba violations | Philly saving | Philly violations |
|---|---|---|---|---|
| S0 ASAP | 0 | 6.9% | 0 | 15.3% |
| S1 greedy | 5.4% (4.2-6.8) | 16.7% | 2.5% (1.0-3.9) | 16.9% |
| S2 threshold | 4.3% (3.2-5.4) | 11.3% | 2.1% (0.8-3.4) | 15.5% |
| S4 simple | 4.3% (3.4-5.3) | 11.0% | 1.7% (0.7-2.7) | 15.1% |
| S3 oracle | 12.3% (10.8-14.0) | 8.7% | 7.1% (2.5-11.6) | 16.7% |

Pool: 11.8 GPUs (Alibaba), 21.4 (Philly). Savings on Philly are about half of Alibaba's; Philly arrivals
are burstier, so even ASAP misses 15% of deadlines.

### E5 alpha tracking (robust MPC S4; 4 weeks, 3 seeds)

| alpha | S4 MPC violations | S4 MPC saving | S4 simple violations |
|---|---|---|---|
| 0.01 | 0.1% | 7.1% | 0.6% |
| 0.05 | 0.2% | 8.2% | 1.1% |
| 0.10 | 0.4% | 8.9% | 0.8% |
| 0.20 | 0.7% | 9.4% | 1.7% |

Shifted scenario (errors 1.6x the calibration level): MPC violations 0.2% to 1.3%, still below alpha.
References: S1 3.1% violations / 10.6% saving; oracle 0% / 15.6%.

### E6 ablations (alpha 0.05)
Full 0.2% violations / 8.2% saving. Without conformal bounds 2.4% / 10.6%. Perfect carbon forecast
0.2% / 10.7%. Perfect energy and duration 0% / 10.8%. Carbon CVaR 0% / 4.1%. "No weather" not run
(F4 not rebuilt on this machine).

### E7 sensitivity (S4 simple / oracle saving)
Load 0.4: slack 6 h 2.4% / 8.0%, 48 h 10.3% / 22.7%. Load 0.8: 0.6-2.9% / 6-11%, with violations of
30-40% for every scheduler. Interruptible share 0 vs 1 changes the oracle (7.7% to 9.2%) but not S1 or
S4 simple, whose results are identical in both cells: worth checking whether those schedulers use pausing.

### E8 marginal-emission proxies
Gas is at or above the merit threshold 95.4% of the time. Oracle saving: average accounting 10.7%,
merit-gas proxy 1.8%, constant-gas proxy -0.4%. S4 simple 4.0% falls to 0.2% (not significant). These are
proxies, not measured marginal data; the demand regression was not run (no demand data).

### E9 cross-zone transfer (train 2018-2022, CQR on 2023, test 2024 to 2026-09)
Skill = MAE / persistence MAE on the target (below 1 is better).
- Raw transfer fails when levels differ: PL to FR 77.95, DE_LU to FR 22.36; as-is coverage 0-6%.
- Scaled transfer (series divided by its own training mean): most pairs below 1, for example GB to
  DE_LU 0.74, ES to PL 0.68; into FR stays above 1 (FR is flat, persistence is hard to beat; FR on FR 0.98).
- Recalibrating CQR on one year of target data brings scaled coverage to 0.85-0.92 for every pair.

## Interpretation
1. The central claim (E5) holds: the robust scheduler keeps the deadline-violation rate below the chosen
   alpha, including under a 1.6x error shift, and E6 shows the conformal bounds are what does it.
2. The carbon savings are modest (Alibaba 4-5%, Philly about 2%, oracle 7-12%) and need slack and spare
   capacity (E7).
3. E8 is the most important caveat: under marginal-emission proxies most of the saving disappears.
4. Forecasters transfer across zones only with level scaling plus recalibration on local data (E9).

## Code changes on this branch
- `sim/traces.py`: `load_philly_jobs` (last attempt, GPUs across machines, "None" timestamps dropped)
  with a test; `GQ_TRACE=alibaba|philly` selects the trace in `e4.py` and in the sweep workers.
- `energy/preprocess.py`: outlier repeats are flagged per (GPU, configuration), need both 3 MAD and a 10%
  deviation, and are now left out of the means (they were only counted before; pooling GPUs flagged whole GPUs).
- `experiments/e9.py`: skill is computed on hours where persistence exists (a gap made it NaN for FR and ES).
- `configs/emission_factors.yaml`: "Hydro Run-of-river and pondage" spelled as ENTSO-E does (value unchanged).
- `configs/gpus.yaml`: RTX 3050 / 4050 / 5070 Laptop entries, marked as assumptions.

## Needs the owner
- Verify the laptop GPU peak-TFLOPS and TDP values (derived, not datasheet; laptop TGP varies by model).
- Decide whether scheduling should use the fitted energy models instead of the synthetic estimator.
- Check why S1 and S4 simple ignore the interruptible share (E7).
- Rebuild F4 / EW to run the E6 "no weather" ablation.
- Review this report and merge the branch.
