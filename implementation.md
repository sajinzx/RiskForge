# GreenQueue — Implementation Guide

**Uncertainty-Aware Carbon Scheduling for AI Workloads**
Owner: Sajin MK PSGTECH

This file is the single source of truth for building GreenQueue. It is written for two readers:

1. **You (the owner)**: so you understand *what* each step does, *why* it exists, and *what you personally must do* (download data, run GPU measurements, review results, commit code).
2. **Claude Code (the implementation assistant)**: so it can build the repo module by module, check its own work, and stop when something needs your decision.

> **Rule zero for Claude Code:** Read this whole file before writing any code. Follow the section "Self-correcting loop" (§13) for every task, and the section "Git and commit rules" (§15) for every commit. Never invent data, results or citations.

---

## Table of contents

1. Project in plain words
2. Research question and contributions
3. Roles: who does what
4. System architecture
5. Tech stack per version
6. Datasets: what to download and how
7. Preprocessing specification
8. Module 1: Carbon forecaster (models, training, tuning)
9. Module 2: Energy estimator (measurement, models, training)
10. Module 3: Workload simulator
11. Module 4: Schedulers S0 to S5
12. Evaluation protocol and experiments E1 to E9
13. Self-correcting loop (for Claude Code)
14. Version roadmap with week-by-week tasks
15. Git and commit rules (no Claude as contributor)
16. Repository layout and engineering standards
17. What to study, phase by phase
18. Risks and what to do when things go wrong
19. Definition of done
20. Glossary

---

## 1. Project in plain words

AI jobs such as training runs, fine-tuning and batch inference consume a lot of electricity. The carbon emitted per kilowatt-hour (the grid's **carbon intensity**, in gCO₂/kWh) changes every hour, depending on how much wind, solar, nuclear, gas or coal is generating at that moment.

Many AI jobs do not need to start immediately; they can wait a few hours. If we start them when the grid is cleaner, we emit less carbon for the same work.

The difficulty: we do not know the future carbon intensity, and we do not know exactly how much energy a job will use. Both are **forecasts**, and both can be wrong.

GreenQueue:

1. **Forecasts carbon intensity** 1 to 48 hours ahead, with a calibrated interval ("we are 90% sure it will be between X and Y").
2. **Estimates job energy** from job features (model, GPU, batch size, etc.), also with a calibrated interval.
3. **Schedules jobs** so total carbon falls, while keeping the share of jobs that miss their deadline under a target you choose (for example 5%).
4. **Evaluates honestly** by replaying real cluster job traces against real historical grid data, and comparing against a perfect-information **oracle** that represents the best possible result.

---

## 2. Research question and contributions

**Research question:** When both the carbon forecast and the energy estimate are wrong, can distribution-free uncertainty (conformal prediction) let a scheduler save carbon while keeping the deadline-miss rate under a chosen target α?

**Intended contributions** (to be re-checked in v1 weeks 1–2):

| # | Contribution | Where it is built |
|---|---|---|
| C1 | Open, reproducible benchmark linking carbon forecasting, energy estimation and trace-replay scheduling | Whole repo |
| C2 | Gray-box (physics-informed) energy model that generalizes to unseen models and GPUs, with calibrated intervals | `energy/` (v2) |
| C3 | Scheduler using adaptive conformal bounds on both uncertainty sources, with an empirical test of deadline violations under shift | `schedulers/s4_robust.py` (v2) |
| C4 | Sensitivity analysis: how much saving survives average vs marginal emissions accounting | `experiments/e8` (v3) |

**Novelty check (your task, weeks 1–2):** search Google Scholar, arXiv and Semantic Scholar for combinations of: *carbon-aware scheduling* + *conformal prediction*; *carbon intensity forecasting* + *uncertainty* + *scheduling*; *GPU energy prediction* + *conformal*. Record every close paper in `paper/related_work.md` with one line on how GreenQueue differs. If a close match exists, narrow the claims to C2 and C4.

---

## 3. Roles: who does what

Three roles. Each task in §14 is tagged with the role that owns it.

### 3.1 Owner (you) — `[OWNER]`

You are the researcher and the only Git contributor. You:

- Register for data accounts and obtain API tokens (ENTSO-E).
- Run the data download scripts on your machine (or tell Claude Code to run them once tokens are in `.env`).
- Run GPU energy measurements on Colab / Kaggle / university GPU (Claude Code cannot access a GPU).
- Make all research decisions: zones, date ranges, α targets, slack distributions, scope freezes.
- Review every diff before it is committed. Write or approve every commit message.
- Read the papers in §17 and write the report.
- Decide when a version is "done" using the exit criteria.

### 3.2 Implementation assistant (Claude Code) — `[CC]`

Claude Code writes code, tests and documentation under your direction. It:

- Implements one task at a time from §14, following the self-correcting loop in §13.
- Writes unit tests before or alongside code, and runs `ruff`, `mypy` and `pytest` after every change.
- Logs progress in `PROGRESS.md` and open questions in `QUESTIONS.md`.
- **Never** fabricates data, results, emission factors or citations. If something is unknown, it writes the question in `QUESTIONS.md` and stops that task.
- **Never** commits with any AI attribution (see §15). Preferably it does not commit at all: it stages changes and proposes a message for you.
- **Never** commits raw data, tokens or `.env`.

### 3.3 Reviewer / guide (faculty supervisor or peer) — `[REVIEW]`

Optional but recommended. Reviews the research question, the novelty check (end of week 2), the v1 report and the v2 experiment design.

### 3.4 Decision gates (Claude Code must stop and ask the owner)

Claude Code must stop and write to `QUESTIONS.md` when:

- A dataset is missing, has a different schema than expected, or has gaps larger than the rules in §7.
- A choice affects research results (emission factors, splits, α, slack distributions, zones).
- A test fails three loop iterations in a row (§13).
- A dependency must be added that is not in §5.
- Anything would delete data or rewrite Git history.

---

## 4. System architecture

### 4.1 The four modules

```
                ┌──────────────────────────┐
 Grid data ───▶ │ 1. CARBON FORECASTER     │──▶ CI forecast + interval [lo, hi]
 Weather   ───▶ │  (carbon/)               │     for horizons 1..48 h
                └──────────────────────────┘
                                                        │
                ┌──────────────────────────┐            ▼
 Job features ▶ │ 2. ENERGY ESTIMATOR      │──▶ ┌──────────────────────┐
 (model, GPU,   │  (energy/)               │    │ 4. SCHEDULER         │
 batch, seq,    └──────────────────────────┘    │  (schedulers/)       │
 duration)            energy + interval ───────▶│  S0..S5              │
                                                │  minimises risk-     │
                ┌──────────────────────────┐    │  adjusted carbon s.t.│
 Job traces ──▶ │ 3. WORKLOAD SIMULATOR    │◀──▶│  deadline constraints│
 Historical CI ▶│  (sim/)                  │    └──────────────────────┘
                │  finite GPU pool, clock  │
                └──────────────────────────┘
                          │
                          ▼
              Metrics: carbon, deadline misses, wait time,
              regret vs oracle, CVaR  ──▶ experiments/ ──▶ paper/
```

### 4.2 Data flow per simulation time step

At every step *t* (30 or 60 minutes):

1. The simulator releases jobs whose arrival time ≤ *t* into the queue.
2. The carbon forecaster is called with information available at *t* only, returning point forecasts and intervals for *t+1 … t+48*.
3. The energy estimator returns energy and duration estimates (point + interval) for each queued job.
4. The scheduler decides which jobs start, pause or resume now (and plans ahead, but only *now* is executed — this is model predictive control).
5. The simulator advances the clock, runs the chosen jobs using **true** energy and **true** carbon intensity, and records emissions, completions and deadline misses.

**Key principle:** the scheduler only ever sees forecasts; the simulator's accounting only ever uses true values. This separation is what makes the evaluation honest. It must be enforced in code: the scheduler interface receives a `ForecastBundle`, never the true carbon series.

### 4.3 Core interfaces (Claude Code implements these first)

```python
# greenqueue/core/types.py  (pydantic or dataclasses)
class Job:          # id, arrival, true_duration_h, gpus, true_energy_kwh, deadline,
                    # interruptible: bool, checkpoint_overhead_h, features: dict
class CarbonForecast:  # issue_time, horizon_times[], point[], lo[], hi[]
class EnergyEstimate:  # job_id, energy_point, energy_lo, energy_hi,
                       # duration_point, duration_hi
class ForecastBundle:  # carbon: CarbonForecast, energy: dict[job_id, EnergyEstimate]
class Decision:        # job_id -> action in {"start","run","pause","wait"}

class CarbonForecaster(Protocol):
    def fit(self, train_df, calib_df) -> None: ...
    def predict(self, issue_time) -> CarbonForecast: ...

class EnergyEstimator(Protocol):
    def fit(self, train_df, calib_df) -> None: ...
    def predict(self, jobs: list[Job]) -> dict[str, EnergyEstimate]: ...

class Scheduler(Protocol):
    def decide(self, t, queue: list[Job], running: list[Job],
               bundle: ForecastBundle, free_gpus: int) -> list[Decision]: ...
```

---

## 5. Tech stack per version

Python **3.11**. Package manager: **uv** (fast, lockfile) — `pip` + `requirements.lock` is an acceptable fallback. Pin every version in the lockfile.

### 5.1 v1 (core)

| Purpose | Library | Why |
|---|---|---|
| Data frames | `pandas`, `pyarrow` | Standard; Parquet storage |
| Numerics | `numpy`, `scipy` | Arrays, curve fitting |
| HTTP | `requests`, `tenacity` | API calls with retry/backoff |
| ENTSO-E client | `entsoe-py` | Official-format queries to ENTSO-E |
| Holidays | `holidays` | Calendar features per country |
| Classical ML | `scikit-learn` | Ridge, pipelines, metrics |
| Gradient boosting | `lightgbm` | Quantile regression for forecasts and energy |
| Tuning | `optuna` | Bayesian hyperparameter search |
| Optimization | `pulp` with `HiGHS` (or `highspy`) | Oracle MILP / LP |
| GPU measurement | `pynvml` (`nvidia-ml-py`), `codecarbon` | Power logging on Colab/Kaggle |
| DL workloads (measurement only) | `torch`, `torchvision`, `transformers`, `accelerate` | The workloads you measure |
| Config | `pyyaml` + `pydantic` | Typed, validated experiment configs |
| Plots | `matplotlib` | Paper-quality figures |
| CLI | `typer` | `greenqueue run e1` style commands |
| Quality | `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit` | Tests and linting |
| Logging | `structlog` or stdlib `logging` | Run logs |

Conformal prediction (split, CQR, ACI) is **implemented by hand** in `carbon/conformal.py` (~150 lines). This is deliberate: you must be able to explain every line. `mapie` may be used only as a cross-check in tests.

### 5.2 v2 (research core) — adds

| Purpose | Library |
|---|---|
| Weather archive | Open-Meteo HTTP API (`requests`, or `openmeteo-requests`) |
| Deep baseline (optional) | `neuralforecast` (N-HiTS / TFT) on top of `torch` |
| Experiment tracking (optional) | `mlflow` local file store, or plain CSV + JSON logs |
| Parallel runs | `joblib` |

### 5.3 v3 (depth and polish) — adds

| Purpose | Library |
|---|---|
| Demo dashboard | `streamlit` + `plotly` |
| Faster sims / benchmarks | `polars` (optional), `numba` (optional) |
| Paper | LaTeX (Overleaf) or `quarto` |

### 5.4 Compute

- Laptop/CPU: everything except energy measurement. LightGBM and MILPs run fine on CPU.
- GPU (Colab / Kaggle / university): energy measurement harness only (`energy/harness/`). Record the exact GPU model, driver and CUDA version for every run.

---

## 6. Datasets: what to download and how

All downloads are scripted in `data/scripts/`. Raw files go to `data/raw/` (git-ignored), cleaned files to `data/processed/` (git-ignored). Only the scripts and a `data/MANIFEST.md` (source, date downloaded, licence, SHA-256 of each file) are committed.

> **Your first-week job `[OWNER]`:** for each source, open its website, read the current terms, licence and rate limits, and write them into `data/MANIFEST.md`. APIs and terms change; do not trust this file blindly.

### 6.1 UK Carbon Intensity API (NESO) — Great Britain

- **What:** half-hourly actual and forecast carbon intensity (gCO₂/kWh), national and regional, plus generation mix.
- **Access:** free, no key. Base URL `https://api.carbonintensity.org.uk`.
- **Endpoints used:** `/intensity/{from}/{to}` (national actual + forecast), `/generation/{from}/{to}` (mix), `/regional/...` (optional).
- **Download plan:** request in windows of at most 14 days, sleep between calls, retry with exponential backoff. Range: 2018-01-01 to the latest complete month.
- **Note:** the API's own `forecast` field is a ready-made **baseline** to beat. Save it.
- **Script:** `data/scripts/download_uk_ci.py --start 2018-01-01 --end 2026-09-30`
- **Who:** `[CC]` writes the script; `[OWNER]` runs it and checks the row count (≈ 48 rows per day).

### 6.2 ENTSO-E Transparency Platform — DE, FR, PL, ES

- **What:** actual generation per production type (fuel), hourly or 15-minute, per bidding zone.
- **Access:** free; register at the Transparency Platform, then request a RESTful API security token (by email to the ENTSO-E transparency team / via account settings — check the current procedure). Put it in `.env` as `ENTSOE_API_KEY=...`. **Never commit `.env`.**
- **Bidding zones (entsoe-py codes):** `DE_LU` (Germany), `FR` (France), `PL` (Poland), `ES` (Spain). GB is **not** taken from ENTSO-E (GB data stops after Brexit); use §6.1.
- **Query:** `EntsoePandasClient.query_generation(country_code, start, end, psr_type=None)`, in monthly chunks.
- **Range:** 2018-01-01 to latest complete month.
- **Who:** `[OWNER]` gets the token; `[CC]` writes `download_entsoe.py`; `[OWNER]` runs it.

**Computing carbon intensity from the mix** (`carbon/intensity.py`):

$$CI(t) = \frac{\sum_f G_f(t)\, EF_f}{\sum_f G_f(t)}$$

where $G_f(t)$ is generation (MW) of fuel type $f$ and $EF_f$ its lifecycle emission factor (gCO₂eq/kWh).

Default emission factors (IPCC AR5 WGIII Annex III lifecycle medians, store in `configs/emission_factors.yaml` with a citation comment). **`[OWNER]` must verify these numbers against the IPCC table before use:**

| ENTSO-E production type | EF (gCO₂eq/kWh) | Source note |
|---|---|---|
| Fossil Hard coal | 820 | IPCC median, coal |
| Fossil Brown coal/Lignite | 820 (sensitivity: ~1050) | IPCC has one coal value; lignite higher in other sources — document choice |
| Fossil Gas | 490 | IPCC median, gas combined cycle |
| Fossil Oil | ~650 | Not in IPCC table; take from a cited secondary source |
| Biomass | 230 | IPCC median (dedicated) |
| Nuclear | 12 | IPCC median |
| Hydro (all types) | 24 | IPCC median |
| Wind Onshore | 11 | IPCC median |
| Wind Offshore | 12 | IPCC median |
| Solar | 48 | IPCC median, utility PV |
| Geothermal | 38 | IPCC median |
| Other / Other renewable / Waste | documented assumption | Flag in limitations |
| Hydro Pumped Storage (consumption) | exclude from denominator | Storage, not generation; document |

Imports/exports are ignored in v1 (production-based intensity). State this as a limitation; consumption-based accounting is a v3 extension.

### 6.3 Open-Meteo — weather covariates (v2)

- **What:** wind speed at 10 m and 100 m, shortwave radiation, temperature at 2 m, cloud cover.
- **Access:** free for non-commercial use, no key.
- **Leakage rule (critical):** use **archived forecasts** as they were issued (Open-Meteo's *Historical Forecast API* and/or *Previous Runs API*), **not** the reanalysis/ERA5 archive. Reanalysis contains the weather that actually happened, which the scheduler could not have known. Check which date range each archived-forecast product covers; this may limit the v2 training period (archived forecasts typically start around 2021–2022).
- **Locations:** 3–5 representative points per zone (e.g. major wind and solar regions), averaged or kept as separate features. List coordinates in `configs/zones.yaml`.
- **Script:** `data/scripts/download_weather.py`.

### 6.4 Public energy-per-request data

| Source | What you take |
|---|---|
| ML.ENERGY Leaderboard / Benchmark | Energy per request for LLMs on known GPUs |
| Hugging Face AI Energy Score | Energy per task/model, standardized setup |
| MLPerf Inference (power results) | System power for standard models |

Small tables. Download manually or by script into `data/raw/energy_public/`. Used for **validation** of the energy model and as extra training rows (flag `source=public`).

### 6.5 Your own energy measurements (core energy dataset)

Produced by you with the harness in §9.1. Stored as `data/raw/energy_own/<gpu>/<date>/*.parquet`. This is the most important dataset you create. **`[OWNER]` runs it; `[CC]` writes the harness.**

### 6.6 Job traces

| Trace | Use | Where |
|---|---|---|
| Alibaba GPU cluster trace (cluster-trace-gpu-v2020, PAI) | Main trace: arrivals, durations, GPU counts | `github.com/alibaba/clusterdata` |
| Microsoft Philly traces | Second trace, training jobs | `github.com/msr-fiddle/philly-traces` |
| Azure Public Dataset (incl. LLM inference traces) | Batch inference arrivals (v3 LLM study) | `github.com/Azure/AzurePublicDataset` |
| SenseTime Helios | Cross-check trace | `github.com/S-Lab-System-Group/HeliosData` |

Traces have **no deadlines and no carbon information**. Deadlines are synthesized (§10.3). Trace timestamps are shifted onto the carbon-data calendar (documented offset; keep weekday alignment).

### 6.7 Expected disk usage

UK + 4 ENTSO-E zones for ~8 years: a few hundred MB as Parquet. Weather: < 1 GB. Traces: Alibaba ~ several GB raw, Philly ~ GB — process to compact Parquet with only needed columns.

---

## 7. Preprocessing specification

All preprocessing lives in `carbon/preprocess.py`, `energy/preprocess.py`, `sim/traces.py`. Every parameter below lives in `configs/preprocess.yaml`, not hard-coded.

### 7.1 Time handling (all data)

| Parameter | Value | Why |
|---|---|---|
| Time zone | **UTC everywhere** internally | Avoids daylight-saving bugs |
| Base resolution | 30 min (GB native); ENTSO-E resampled to 60 min; simulator `dt` = 60 min default, 30 min in sensitivity test | One common grid |
| Resampling | Mean of power (MW) within interval; intensity recomputed after resampling, never averaged directly from sub-intervals with different totals | Correct energy weighting |
| Timestamp convention | Interval **start** labels | Consistent joins |
| Daylight-saving duplicates | Handled by UTC conversion; assert no duplicate index | Correctness |

### 7.2 Missing data and outliers (carbon/generation)

| Rule | Value |
|---|---|
| Gap ≤ 2 h | Linear interpolation, flag `imputed=1` |
| Gap > 2 h and ≤ 24 h | Seasonal fill from same hour previous week, flag `imputed=2` |
| Gap > 24 h | Leave missing; rows excluded from training targets and from simulator test windows; report count |
| Negative generation values | Set to 0 (except storage consumption which is excluded) |
| Outliers | Values beyond physical bounds (CI < 0 or > 1200 g/kWh) → missing → rules above |
| Reporting | Script prints and saves `data/processed/quality_report_<zone>.json` (missing %, imputed %, outliers) |

### 7.3 Forecasting features (carbon)

Forecast **issue times**: 00:00 and 12:00 UTC daily. Horizons h = 1…48 hours.

| Feature group | Features | Notes |
|---|---|---|
| Lags of CI | t−1, t−2, t−3, t−6, t−12, t−24, t−48, t−168 h (relative to **issue time**) | Only values known at issue time |
| Rolling stats | Mean and std of CI over last 24 h and 168 h | Computed up to issue time |
| Calendar (target time) | Hour of day (sin/cos), day of week (one-hot or categorical), month (sin/cos), holiday flag (per country) | Known in advance, no leakage |
| Horizon | h as a feature (single global model) | Alternative: one model per horizon bucket {1–6, 7–12, 13–24, 25–48} — try both, pick on calibration set |
| Generation mix lags (v2) | Share of wind, solar, gas, coal at issue time | ENTSO-E zones |
| Weather (v2) | Forecasted wind speed (100 m), shortwave radiation, temperature, cloud cover **for the target time, from the forecast issued at or before the issue time** | Leakage rule |
| Baseline forecast (GB) | NESO's own forecast for the target time | Strong feature + baseline |

**Target:** CI at target time (gCO₂/kWh), no transform for LightGBM; standardized (z-score fitted on train only) for ridge.

**Leakage tests `[CC]` must write:** for a random sample of rows, assert that every feature timestamp ≤ issue time (except calendar features and weather forecasts whose *issue* timestamp ≤ issue time).

### 7.4 Splits (chronological, never shuffled)

Default (adjust to actual data availability, record final choice in `configs/splits.yaml`):

| Split | Period | Use |
|---|---|---|
| Train | 2018-01-01 → 2022-12-31 | Fit models |
| Calibration | 2023-01-01 → 2023-12-31 | Conformal calibration, model selection |
| Test | 2024-01-01 → latest | Final numbers, touched once per version |

For weather-covariate models (v2) the train start moves to when archived forecasts begin. Compare with-weather and without-weather models **on the same test period**.

**Rolling-origin evaluation:** inside the test period, refit (or update) monthly using all data up to that month; report metrics per month. Hyperparameter tuning uses rolling-origin folds **inside train only** (e.g. 4 folds, each validating on a 3-month block following its training window).

### 7.5 Energy data preprocessing

| Step | Rule |
|---|---|
| Power sampling | NVML every 100 ms (configurable) |
| Idle subtraction | Measure 60 s idle before each run; subtract mean idle power |
| Warm-up | Discard first 30 s or first N iterations (configurable) |
| Energy | Trapezoidal integration of (power − idle) over the measured window, in kWh |
| Normalization | Also store energy per sample / per token / per step |
| Repeats | ≥ 3 repeats per config (5 preferred); keep all, also store mean and std |
| Outliers | Repeats with energy > 3 MAD from the config median are flagged, not deleted |
| Features | model name, model family, parameter count, FLOPs per sample (computed with `fvcore`/`torch.utils.flop_counter` or formula), batch size, sequence length, precision (fp32/fp16/bf16), GPU type, GPU TDP, mean utilization, duration |
| Targets | `energy_kwh`, `duration_s` |

### 7.6 Trace preprocessing

| Step | Rule |
|---|---|
| Keep jobs | Training, fine-tuning, batch inference with duration ≥ 10 min |
| Drop | Interactive/online jobs, jobs with missing duration, failed jobs (keep a count) |
| GPU count | From trace; cap at pool size |
| Duration | From trace (true duration); rounded up to simulator `dt` |
| Power per GPU | Sampled from your measured power distribution for the matched job class (documented mapping) |
| Time alignment | Shift trace start to the start of the test period; keep weekday alignment |
| Subsampling | Optional to reach target cluster utilization (40%, 60%, 80%) |

---

## 8. Module 1: Carbon forecaster

### 8.1 What it does (for your understanding)

At each issue time, it predicts the carbon intensity for each of the next 48 hours, and gives an interval that should contain the true value 90% of the time. If intervals are too narrow, the scheduler is overconfident and misses deadlines or emits more than expected. If too wide, the scheduler becomes timid and saves little carbon.

### 8.2 Model ladder (build in this order; each must beat the previous to justify complexity)

| ID | Model | Version | Notes |
|---|---|---|---|
| F0 | Persistence: CI(t+h) = CI(issue time) | v1 | Simplest baseline |
| F1 | Seasonal naive: CI(t+h) = CI(t+h−24) (and −168 variant) | v1 | Captures daily cycle |
| F2 | Ridge regression on lags + calendar | v1 | Linear baseline |
| F3 | LightGBM quantile regression (lags + calendar) | v1 | Main v1 model |
| F4 | F3 + weather covariates + mix features | v2 | Main v2 model |
| F5 | N-HiTS or TFT (`neuralforecast`) | v2 optional | Deep baseline only |
| FN | NESO official forecast (GB only) | v1 | External baseline |

### 8.3 Uncertainty methods

| ID | Method | Version | How it works |
|---|---|---|---|
| U0 | Raw quantiles | v1 | LightGBM trained at τ = 0.05 and 0.95 (and 0.5 for median) |
| U1 | Split conformal on point model | v1 | Absolute residuals on calibration set; interval = point ± their (1−α) quantile, per horizon bucket |
| U2 | CQR | v1 | Conformity score s = max(lo − y, y − hi); widen [lo, hi] by the (1−α)(1+1/n) quantile of s on calibration set, per horizon bucket |
| U3 | ACI | v2 | Online update of the miscoverage level: α_{t+1} = α_t + γ(α − err_t), where err_t = 1 if y_t outside interval. Applied on top of CQR scores |

Target coverage: 1−α = 0.90 by default; also report 0.80 and 0.95.

ACI note: errors are only observed after the target time passes, so the update for horizon h uses the outcome observed h hours later (delayed feedback). Implement one α_t per horizon bucket.

### 8.4 Training procedure

1. Build features with `carbon/features.py` for each zone → `data/processed/features_<zone>.parquet`.
2. For each model in the ladder: tune on rolling-origin folds inside train (Optuna, below), refit on full train with best params.
3. Calibrate conformal methods on the calibration split.
4. Evaluate on test with rolling origin; save predictions to `results/forecasts/<zone>/<model>_<uq>.parquet`.

### 8.5 Hyperparameter tuning

Objective: mean pinball loss across τ ∈ {0.05, 0.5, 0.95} on validation folds. Optuna TPE sampler, seed fixed, **50 trials** per zone (v1), **100** (v2). Median pruner after 2 folds.

**LightGBM quantile (F3/F4):**

| Parameter | Range |
|---|---|
| `num_leaves` | 15 – 255 (int) |
| `learning_rate` | 0.01 – 0.1 (log) |
| `n_estimators` | up to 3000 with early stopping (100 rounds) on fold validation |
| `min_child_samples` | 10 – 200 |
| `feature_fraction` | 0.5 – 1.0 |
| `bagging_fraction` | 0.5 – 1.0, `bagging_freq` = 1 |
| `lambda_l2` | 1e-3 – 10 (log) |
| `max_bin` | 255 (fixed) |

One model per quantile τ with the same tuned structure (tune on τ=0.5, then reuse).

**Ridge (F2):** `alpha` ∈ [1e-3, 1e3] log grid (13 values), chosen by fold validation.

**ACI (U3):** γ ∈ {0.001, 0.005, 0.01, 0.05}, chosen on the calibration period by long-run coverage closest to target with smallest mean width.

**N-HiTS (F5, optional):** input size 168, horizon 48, max steps 1000, learning rate 1e-3, early stopping; small grid only — it is a baseline, not the contribution.

### 8.6 Metrics

Pinball loss (per τ, per horizon), MAE of median, empirical coverage, mean interval width, **rolling 30-day coverage over time** (to expose regime changes), coverage by horizon bucket.

---

## 9. Module 2: Energy estimator

### 9.1 Measurement harness (`energy/harness/`) — you run this on a GPU

**Workload sweep:**

| Workload | Configurations |
|---|---|
| Small CNN (ResNet-18 on CIFAR-10) | batch ∈ {32, 64, 128, 256}; fp32/fp16 |
| Vision Transformer (ViT-B/16, ImageNet subset or synthetic) | batch ∈ {16, 32, 64} |
| BERT-base fine-tune (e.g. SST-2) | batch ∈ {8, 16, 32}; seq ∈ {128, 256, 512} |
| LLM inference 1B–3B (e.g. TinyLlama-1.1B, Qwen2.5-1.5B, Llama-3.2-1B/3B — check licences) | batch ∈ {1, 4, 8, 16}; prompt ∈ {128, 512}; gen tokens ∈ {64, 256}; fp16 |

Each config: run for a fixed number of steps (or ≥ 2 minutes), 5 repeats, idle measured before. The harness writes one Parquet row per run with all features in §7.5 and a raw power trace file.

**Your steps `[OWNER]`:**
1. Open `energy/harness/colab_runner.ipynb` on Colab or Kaggle with a GPU.
2. Run the cell that prints GPU name, driver, CUDA — note it.
3. Run the sweep (it saves progress so a disconnected session can resume).
4. Download the output folder to `data/raw/energy_own/<gpu>/<date>/`.
5. Repeat on as many GPU types as you can access (T4, P100, L4, A100 …). Hardware variety is the main limit of this dataset; state it in limitations.

### 9.2 Models

| ID | Model | Version |
|---|---|---|
| EN0 | Linear regression on log-features (log FLOPs, log batch, GPU one-hot) | v1 |
| EN1 | LightGBM (black box) | v1 |
| EN2 | Gray-box physics model | v2 |

**Gray-box model (EN2):**

- Time: $T = \dfrac{\text{FLOPs}_{\text{total}}}{\text{PeakFLOPs}_{gpu}\cdot \eta(b)}$, with efficiency $\eta(b) = \eta_{\max}\,\dfrac{b}{b + b_{1/2}}$ (saturates with batch size *b*).
- Power above idle: $P = (P_{\max,gpu} - P_{\text{idle},gpu})\cdot u^{k}$, where utilization $u$ is approximated by $\eta(b)/\eta_{\max}$.
- Energy: $E = P \cdot T$.
- Free parameters per GPU type: $\eta_{\max}, b_{1/2}, k$ (3 numbers), fitted with `scipy.optimize.least_squares` on log-energy residuals. Peak FLOPs and TDP come from vendor spec sheets (cite in `configs/gpus.yaml`).
- For an **unseen GPU**: use parameters pooled across seen GPUs, scaled by spec-sheet values. This is why gray-box should generalize better than black box.

### 9.3 Validation

- Random K-fold (K=5) — optimistic reference.
- **Leave-one-model-out (LOMO)** — hold out one workload entirely.
- **Leave-one-GPU-out (LOGO)** — hold out one GPU type entirely.
- Report the gap between random-split and leave-one-out MAPE. A big gap means the model memorizes rather than generalizes.

### 9.4 Uncertainty

Split conformal on log-residuals |log y − log ŷ|, **group-conditional** (Mondovian): separate calibration quantiles per model family and per GPU type; fall back to pooled quantile when a group has < 20 calibration points. Interval on energy and on duration (duration upper bound is what protects deadlines).

### 9.5 Hyperparameter tuning

- EN0: none (or ridge alpha grid as in §8.5).
- EN1 LightGBM: Optuna 50 trials, objective = mean LOMO MAPE; ranges as §8.5 but `num_leaves` 7–63 and `min_child_samples` 5–50 (small dataset).
- EN2: no tuning beyond least-squares fit; bounds η_max ∈ (0,1], b_{1/2} > 0, k ∈ [0.5, 3].

### 9.6 Metrics

MAPE, median absolute percentage error, coverage, interval width (relative), LOMO/LOGO gap.

---

## 10. Module 3: Workload simulator

### 10.1 What it does

A discrete-time simulator of a finite GPU cluster. It is the "world": it holds true carbon, true energy and true durations, executes decisions and counts emissions. Its accounting must be exactly right, so it is unit-tested on toy cases where the answer can be computed by hand.

### 10.2 Job model

`arrival, duration (true), gpus, power_per_gpu (true), deadline, interruptible, checkpoint_overhead`.
Classes: batch training, fine-tuning, batch inference. Online inference is excluded (not shiftable).

### 10.3 Synthesized deadlines

| Parameter | Default | Sensitivity values |
|---|---|---|
| Slack (deadline − arrival − duration) | Categorical {6, 12, 24, 48} h with probs {0.3, 0.3, 0.25, 0.15} | All-6, all-12, all-24, all-48; lognormal alternative |
| Interruptible share | 50% | 0%, 100% |
| Checkpoint overhead | 5 min per pause/resume | 0, 15 min |
| Seeds | 5 job-mix seeds | — |

All values in `configs/sim.yaml`; every result file stores the config hash.

### 10.4 Capacity and clock

- GPU pool size: chosen so trace utilization ≈ 60% under ASAP (vary 40/60/80% in E7).
- Time step `dt`: 60 min default (30 min sensitivity).
- Replay length: ≥ 3 months of test period per run.
- Queue discipline when capacity binds: scheduler decides; simulator rejects decisions that exceed capacity and logs an error (this should never happen — it is a test).

### 10.5 Accounting (must be exact)

Emissions per step = Σ over running jobs of (gpus × power_per_gpu × fraction of step running) × CI(t) × dt. A job that misses its deadline still runs to completion (emissions counted) and is counted as a violation. Checkpoint overhead consumes time and energy.

**Toy tests `[CC]`:** 1 job, constant CI → emissions = E × CI; 2 jobs, 1 GPU → second waits; pause/resume adds overhead exactly once per pause; deadline exactly met is not a violation.

---

## 11. Module 4: Schedulers

| ID | Scheduler | Version | What it does |
|---|---|---|---|
| S0 | ASAP | v1 | Start every job as soon as GPUs are free. Baseline for savings. |
| S1 | Greedy window | v1 | Pick the contiguous window inside the slack with lowest **forecast** mean CI. Common published approach. |
| S2 | Threshold | v1 | Run when forecast CI is below its rolling q-quantile (q tuned on calibration period), or when slack runs out. |
| S3 | Oracle | v1 | Offline MILP with **true** CI and **true** energy/duration over the whole horizon. Upper bound. |
| S4 | Robust conformal MPC | v1 simple, v2 full | See below. Main contribution. |
| S5 | RL agent (PPO) | optional | Extra baseline only. |

### 11.1 MILP formulation (shared by S3 and S4)

Variables: $x_{j,t} \in \{0,1\}$ = job *j* runs in step *t*; $s_{j,t}\in\{0,1\}$ = job *j* (re)starts at *t* (for checkpoint overhead).

$$\min \sum_{j,t} x_{j,t}\, \hat E_{j}\, \widehat{CI}_t$$

subject to:
- Work: $\sum_t x_{j,t} \ge \hat D_j$ (steps needed, plus overhead per restart).
- Deadline: $x_{j,t}=0$ for $t \ge$ deadline_j; and $x_{j,t}=0$ for $t <$ arrival_j.
- Capacity: $\sum_j g_j\, x_{j,t} \le G$ for all *t*.
- Non-interruptible jobs: runs contiguous (once started, continue until done) — encode with start variables.
- Restart counting: $s_{j,t} \ge x_{j,t} - x_{j,t-1}$.

**S3 (oracle):** $\hat E, \widehat{CI}, \hat D$ = true values; solved once offline per replay window (split into overlapping 1-week windows if too large; document).

**S4 (robust):**
- $\widehat{CI}_t$ = upper conformal bound (or CVaR-style mean of the upper tail) of the carbon forecast.
- $\hat E_j$ = upper conformal bound on energy.
- $\hat D_j$ = **upper conformal bound on duration** at level 1−α → this is what targets deadline-miss rate ≤ α.
- **MPC:** re-solve every step with the latest forecast over a 48 h lookahead; execute only the decisions for the current step.
- Solver: HiGHS via PuLP, time limit 10 s per solve, MIP gap 1%; if the time limit hits, fall back to S1 decision for that step and log it.
- v1 "simple S4": point CI forecast + conformal duration bound only, no MPC re-solve for running jobs.

### 11.2 Scheduler hyperparameters

| Parameter | Default | Tuned how |
|---|---|---|
| α (deadline target) | 0.05 | Swept {0.01, 0.05, 0.1, 0.2} in E5 (not tuned — it is the user's choice) |
| CI risk level | upper 0.9 quantile | Compare point / q0.9 / CVaR0.9 in E6 |
| S2 threshold q | 0.3 | Grid {0.1…0.5} on calibration period |
| MPC lookahead | 48 h | 24 h sensitivity |

---

## 12. Evaluation protocol and experiments

**Rules:** chronological splits only; test period touched only for final numbers; every number has a block-bootstrap 95% CI (block length 7 days, 1000 resamples); 5 job-mix seeds; every negative result reported.

**Scheduling metrics:** carbon saved vs S0 (% and tCO₂), share of oracle saving captured = (S0 − Sx)/(S0 − S3), deadline-violation rate vs α, mean and p95 wait time, regret vs oracle, CVaR₀.₉ of per-job carbon.

| ID | Question | Version | Main figure |
|---|---|---|---|
| E1 | Forecast accuracy and calibration per zone | v1 | Pinball loss table; coverage vs horizon |
| E2 | Coverage during regime shifts (low-wind weeks, heat waves, seasons) | v2 | Rolling coverage over time, CQR vs ACI |
| E3 | Energy model generalization | v2 | MAPE random vs LOMO vs LOGO |
| E4 | Carbon saved by S1–S4 vs S0, share of oracle captured | v1 | Bar chart with CIs |
| E5 | Does violation rate track α? | v2 | α vs realized violations, diagonal reference |
| E6 | Ablations: no weather, no conformal, no energy uncertainty, perfect energy | v2 | Ablation table |
| E7 | Sensitivity to slack, utilization, zone | v2 | Heatmaps |
| E8 | Average vs marginal emissions proxies | v3 | Saving under each accounting |
| E9 | Cross-zone transfer | v2 | Train-zone × test-zone matrix |

Each experiment = one YAML file in `experiments/configs/eNN.yaml` and one command: `greenqueue run eNN`. Output: `results/eNN/` (tables as CSV, figures as PDF+PNG, a `summary.json`, and the config hash).

**E8 marginal proxy (v3):** since true marginal emissions data is hard to obtain, use documented proxies, e.g. (a) emission factor of the marginal fossil fuel type (gas or coal, by merit-order rule), (b) regression of ΔEmissions on ΔDemand per hour of day. State clearly these are proxies.

---

## 13. Self-correcting loop (for Claude Code)

Claude Code follows this loop for **every** task in §14. The loop is how the work stays correct without you watching each line.

```
┌──────────────────────────────────────────────────────────────────┐
│ 0. READ    implementation.md section for the task, PROGRESS.md,  │
│            QUESTIONS.md, relevant existing code and tests.       │
│ 1. PLAN    Write a 3–8 line plan + acceptance criteria into      │
│            PROGRESS.md under the task ID.                        │
│ 2. TEST    Write/extend tests that encode the acceptance         │
│            criteria (toy cases, leakage checks, shapes).         │
│ 3. BUILD   Implement the smallest change that can pass.          │
│ 4. CHECK   Run:  ruff check . && ruff format --check .           │
│                  mypy greenqueue                                 │
│                  pytest -q                                       │
│                  (task-specific smoke run, e.g. greenqueue run   │
│                   e1 --smoke)                                    │
│ 5. VERIFY  Compare outputs with acceptance criteria and sanity   │
│            bounds (below). Read produced figures/tables.         │
│ 6. DECIDE  ├─ all pass → mark task done in PROGRESS.md, stage    │
│            │   changes, propose commit message (§15). STOP.      │
│            ├─ fail, attempt < 3 → write diagnosis (what failed,  │
│            │   hypothesis, fix) in PROGRESS.md → go to 3.        │
│            └─ fail, attempt = 3, or a decision gate (§3.4) hit → │
│                write to QUESTIONS.md, revert half-done edits,    │
│                STOP and wait for the owner.                      │
└──────────────────────────────────────────────────────────────────┘
```

### 13.1 Sanity bounds (automatic red flags)

Claude Code treats any of these as a failure even if tests pass:

- Carbon intensity outside [0, 1200] gCO₂/kWh.
- Forecast beats the oracle-like "perfect" reference, or test coverage is exactly 100% / 0% → likely leakage or bug.
- Any scheduler saves more carbon than S3 (oracle) on the same replay → bug.
- S0 emissions differ between two runs with the same seed → nondeterminism bug.
- Energy per run outside [0.5×, 2×] of power×time computed independently.
- Conformal coverage on synthetic i.i.d. data differs from target by > 2 percentage points with n ≥ 2000.
- A test-period file is read during tuning (enforced by a guard function `assert_not_test(df)`).

### 13.2 Files the loop maintains

- `PROGRESS.md` — per task: plan, acceptance criteria, attempts, diagnosis, final status.
- `QUESTIONS.md` — open decisions for the owner, each with options and a recommendation.
- `CLAUDE.md` — short project memory for Claude Code: points to this file, lists commands, restates §15. Create it in task T0.1.

### 13.3 Definition of a "task" acceptance criterion

Each must be checkable by a command, e.g. "`pytest tests/sim/test_accounting.py` passes and covers 4 toy cases", "`greenqueue run e1 --smoke` produces `results/e1/coverage.png`", "leakage test passes for 1000 sampled rows".

---

## 14. Version roadmap with tasks

Tags: `[OWNER]` you, `[CC]` Claude Code, `[REVIEW]` supervisor. Each task ID is used in `PROGRESS.md` and in commit messages.

### v1 — Solid core (≈ 12 weeks)

**Scope freeze:** zones GB + DE; forecaster ladder F0–F3 + FN with U0–U2; energy harness + EN0/EN1; simulator; S0, S1, S3 (+ simple S4 if time). Experiments E1, E4.

| Week | Task | Who | What happens and why |
|---|---|---|---|
| 1 | T0.1 Repo skeleton, `pyproject.toml`, uv lock, ruff/mypy/pytest/pre-commit, `CLAUDE.md`, commit-msg hook (§15) | CC | Sets up a clean, reproducible base so every later step is testable. |
| 1 | T0.2 Set Git identity, install hook, verify no AI trailer | OWNER | Ensures you are the only contributor. |
| 1 | T0.3 Literature/novelty check, start `paper/related_work.md` | OWNER | Confirms what is actually new before investing months. |
| 1 | T0.4 Register ENTSO-E, get token, fill `data/MANIFEST.md` with terms | OWNER | Data access takes days; start now. |
| 1–2 | T1.1 UK API downloader + tests (mocked HTTP) | CC | Get GB carbon data and NESO's forecast. |
| 1–2 | T1.2 ENTSO-E downloader + intensity computation + emission-factor config | CC | Turn generation mix into carbon intensity. |
| 2 | T1.3 Run downloads, check quality reports | OWNER | You confirm the data is complete and sensible. |
| 2 | T1.4 Reproduce a simple published shifting result (e.g. "Let's Wait Awhile" style: savings from shifting fixed-length jobs within a window) on GB data | CC + OWNER | Checks your pipeline gives numbers in the same ballpark as the literature. |
| 2 | T1.5 Novelty review | REVIEW | Supervisor signs off on claims. |
| 3–4 | T2.1 Feature builder + leakage tests | CC | Features that only use past information. |
| 3–4 | T2.2 F0, F1, F2, FN + metrics module (pinball, coverage, width) | CC | Baselines everything else must beat. |
| 3–4 | T2.3 F3 LightGBM quantile + Optuna tuning | CC | Main v1 forecaster. |
| 4 | T2.4 Split conformal + CQR (`carbon/conformal.py`) + synthetic coverage tests | CC | Makes intervals trustworthy. |
| 4 | T2.5 Review E1 smoke results, read CQR paper | OWNER | You should be able to explain why coverage is ~90%. |
| 5–6 | T3.1 Measurement harness (NVML logger, workload runners, resumable sweep, Colab notebook) | CC | Tool that turns GPU runs into an energy dataset. |
| 5–6 | T3.2 Run sweep on ≥ 1 GPU type (≥ 2 preferred) | OWNER | Creates your core energy dataset. |
| 6 | T3.3 EN0, EN1 + LOMO/LOGO CV + split conformal for energy/duration | CC | First energy estimates with intervals. |
| 7–8 | T4.1 Core types and simulator + exact-accounting toy tests | CC | The honest "world" for evaluation. |
| 7–8 | T4.2 S0, S1, S2 | CC | Baselines. |
| 7–8 | T4.3 S3 oracle MILP (PuLP + HiGHS), windowing for large problems | CC | Upper bound. |
| 8 | T4.4 Simple S4 (if time) | CC | Early version of the contribution. |
| 9–10 | T5.1 Trace loaders (Alibaba, Philly), deadline synthesis | CC | Realistic job arrivals. |
| 9–10 | T5.2 Download traces | OWNER | Large files; your machine. |
| 9–10 | T5.3 E1 and E4 full runs, bootstrap CIs | CC | First real results. |
| 10 | T5.4 Interpret results, sanity review | OWNER | Decide if numbers make sense. |
| 11–12 | T6.1 `greenqueue reproduce v1` one-command script, README (results → method → limitations) | CC | Reproducibility exit criterion. |
| 11–12 | T6.2 v1 report (4–6 pages): question, data, methods, E1/E4, limitations | OWNER | Your write-up. |
| 12 | T6.3 Report review | REVIEW | |

**v1 exit criteria:** one command reproduces all v1 figures; calibration plot; savings-vs-oracle plot; short report with limitations.

### v2 — Research core

**Scope freeze:** five zones (GB, DE, FR, PL, ES); F4 weather model; U3 ACI; EN2 gray-box with LOMO/LOGO; full S4 with MPC; experiments E2, E3, E5, E6, E7, E9.

| Task | Who | Purpose |
|---|---|---|
| T7.1 Weather downloader (archived forecasts) + leakage tests | CC | Weather helps predict wind/solar-driven CI. |
| T7.2 Download weather; confirm archived-forecast date coverage | OWNER | Determines v2 train period. |
| T7.3 F4 + optional F5 | CC | Better forecasts. |
| T7.4 ACI with delayed feedback | CC | Coverage under distribution shift. |
| T8.1 More GPU measurements (more GPU types) | OWNER | Needed for LOGO to mean anything. |
| T8.2 EN2 gray-box + group-conditional conformal | CC | Contribution C2. |
| T9.1 Full S4 (robust bounds + MPC) | CC | Contribution C3. |
| T9.2 E5 α-tracking experiment | CC | The central research test. |
| T10.1 E2, E3, E6, E7, E9 | CC | Evidence on which parts matter. |
| T10.2 Interpret, write v2 section | OWNER | |

**v2 exit criteria:** evidence on whether violations track α; ablations showing which components matter.

### v3 — Depth and polish

| Task | Who |
|---|---|
| T11.1 E8 marginal-emissions sensitivity | CC + OWNER |
| T11.2 Optional India analysis (only if good hourly generation data is found — e.g. Grid-India reports; check availability and terms) | OWNER decides |
| T11.3 Optional LLM inference trace study (Azure LLM traces) | CC |
| T11.4 Optional decision-latency benchmark (thousands of queued jobs, ms responses; may need greedy/LP relaxation) | CC |
| T11.5 Streamlit demo dashboard | CC |
| T11.6 Paper-style write-up (6–8 pages) | OWNER |

### v4 (optional) — Theory

Conditions under which conformal coverage on duration carries through to the scheduler's deadline guarantee (e.g. exchangeability, capacity not binding), and counter-examples where it breaks (queueing delays add risk not covered by duration bounds). Owner-led.

---

## 15. Git and commit rules (no Claude as contributor)

GitHub lists a person as a contributor if they are the **author** of a commit, or appear in a `Co-Authored-By:` trailer. To keep only you as the contributor:

### 15.1 One-time setup `[OWNER]`

```bash
git config user.name  "Sajin MK"
git config user.email "<your GitHub email or noreply address>"
```

### 15.2 Disable Claude Code's co-author trailer

Create `.claude/settings.json` in the repo (commit it):

```json
{
  "includeCoAuthoredBy": false
}
```

Newer Claude Code versions may use an `attribution` setting instead (e.g. setting commit and PR attribution to empty strings). Check the current Claude Code settings documentation and use whichever key your version supports — you can set both.

### 15.3 Safety net: a commit-msg hook that rejects AI attribution

`scripts/hooks/commit-msg` (installed by `pre-commit` or `git config core.hooksPath scripts/hooks`):

```bash
#!/usr/bin/env bash
# Reject commits that mention AI attribution.
msg_file="$1"
if grep -Eiq 'co-authored-by|claude|anthropic|generated with|ai-generated|🤖' "$msg_file"; then
  echo "ERROR: commit message contains AI attribution. Remove it and retry." >&2
  exit 1
fi
```

Make it executable: `chmod +x scripts/hooks/commit-msg`.

Also check the author before pushing:

```bash
git log --format='%an <%ae>%n%b' origin/main..HEAD | grep -Ei 'claude|anthropic|co-authored' && echo "FIX BEFORE PUSH"
```

### 15.4 Rules for Claude Code (also copied into `CLAUDE.md`)

1. Commit once per completed task step (owner instruction), using the owner's configured identity. Do not `git push` unless the owner explicitly asks.
2. Commit messages must **never** include `Co-Authored-By`, "Claude", "Anthropic", "generated with", emoji robots, or any AI mention. They state only what was done.
3. Never change `git config user.*`.
4. Never commit `.env`, `data/raw/`, `data/processed/`, large result files, or notebook outputs with secrets.
5. Never rewrite history (`rebase -i`, `push --force`) without the owner's explicit instruction.

### 15.5 Commit message format (Conventional Commits)

```
<type>(<scope>): <short imperative summary>   ← ≤ 72 chars

<optional body: what and why, wrapped at 72>
Refs: T2.3
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`, `data`, `exp`.
Examples:
- `feat(carbon): add CQR calibration per horizon bucket`
- `test(sim): add exact accounting toy cases`
- `exp(e4): run greedy and oracle on GB test period`

### 15.6 Branching

`main` stays runnable. One branch per task: `t2.3-lightgbm-quantile`. Merge after `pytest` passes and you have reviewed the diff.

---

## 16. Repository layout and engineering standards

```
greenqueue/
├── implementation.md        # this file
├── CLAUDE.md                # short memory for Claude Code
├── PROGRESS.md              # task log (self-correcting loop)
├── QUESTIONS.md             # open decisions for the owner
├── README.md                # results first, then method, then limitations
├── pyproject.toml, uv.lock
├── .env.example             # ENTSOE_API_KEY=   (real .env git-ignored)
├── .claude/settings.json    # attribution disabled
├── configs/                 # zones.yaml, emission_factors.yaml, gpus.yaml,
│                            # preprocess.yaml, splits.yaml, sim.yaml
├── data/
│   ├── scripts/             # download scripts only
│   ├── MANIFEST.md          # sources, licences, dates, hashes
│   ├── raw/        (ignored)
│   └── processed/  (ignored)
├── greenqueue/
│   ├── core/                # types, interfaces, guards (assert_not_test)
│   ├── carbon/              # ingestion, intensity, features, forecasters, conformal
│   ├── energy/              # harness/, preprocess, models (linear, gbm, graybox), conformal
│   ├── sim/                 # simulator, traces, deadlines, accounting
│   ├── schedulers/          # s0_asap, s1_greedy, s2_threshold, s3_oracle, s4_robust, s5_rl
│   ├── eval/                # metrics, bootstrap, plots
│   └── cli.py               # typer entry point: greenqueue run eNN
├── experiments/configs/     # e1.yaml … e9.yaml
├── results/       (ignored except small summaries)
├── notebooks/               # exploration only; never imported by code
├── paper/                   # related_work.md, report, figures
├── scripts/hooks/commit-msg
└── tests/                   # mirrors greenqueue/ package
```

Standards:
- Fixed seeds everywhere (`configs/seeds.yaml`); seeds recorded in every result.
- One command per figure; `greenqueue reproduce v1|v2|v3`.
- Type hints on all public functions; docstrings that explain *why*, not just *what*.
- Unit tests: simulator accounting (exact), conformal coverage on synthetic data, leakage guards, downloader parsing (mocked HTTP).
- No notebook is a dependency of the pipeline.

---

## 17. What to study, phase by phase

You should be able to explain every formula in the repo without notes. Study each topic **before** reviewing the related task.

| Phase | Topic | What to understand | Reading |
|---|---|---|---|
| Weeks 1–2 | Carbon-aware computing | Temporal vs spatial shifting; why savings depend on slack and grid mix; limits of shifting | Wiesner et al. 2021; Radovanovic et al. 2022; Sukprasert et al. 2024 |
| Weeks 1–2 | Grid carbon accounting | Average vs marginal emissions; production vs consumption-based; lifecycle emission factors | IPCC AR5 WGIII Annex III; Electricity Maps methodology pages |
| Weeks 3–4 | Time-series forecasting | Direct multi-horizon forecasting, lags, calendar effects, leakage, rolling-origin evaluation | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free online) chs. on evaluation |
| Weeks 3–4 | Quantile regression | Pinball loss and why it estimates quantiles | LightGBM docs (quantile objective); Koenker basics |
| Week 4 | Conformal prediction | Split conformal, exchangeability, finite-sample guarantee, CQR | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*; Romano et al. 2019 |
| Weeks 5–6 | GPU energy | Power vs energy, idle power, utilization, FLOPs, why batch size changes efficiency | Zeus (You et al. 2023); Dodge et al. 2022; Patterson et al. 2021; NVML docs |
| Weeks 7–8 | Optimization | LP vs MILP, binary variables, capacity constraints, solver time limits | PuLP docs; any intro OR text (e.g. Hillier & Lieberman chapters on IP) |
| Weeks 9–10 | Statistics for evaluation | Block bootstrap for dependent data, confidence intervals, seeds | Künsch 1989 (block bootstrap) or a tutorial |
| v2 | Distribution shift | ACI update rule and its long-run coverage guarantee | Gibbs & Candès 2021 |
| v2 | Robust optimization / MPC | Why plan ahead but execute one step; chance constraints; CVaR | Rockafellar & Uryasev (CVaR); MPC tutorial |
| v2 | Group-conditional conformal | Mondrian conformal, coverage per group | Vovk et al.; Angelopoulos & Bates §4 |
| v3 | Marginal emissions | Marginal emission factors and how they're estimated | Search recent literature; verify each citation |

> Verify every citation before relying on it (title, venue, year).

**Your understanding checkpoint after each version:** write half a page in `paper/notes_vN.md` answering: What does each module output? Why is the split chronological? Why does conformal give ~90% coverage? Why can't any scheduler beat S3? If you can't answer, pause and study before moving on.

---

## 18. Risks and what to do

| Risk | Mitigation |
|---|---|
| Energy dataset too small | Gray-box model; report LOMO/LOGO honestly; state hardware limits |
| ENTSO-E token delayed | Start with GB only; ENTSO-E zones join when token arrives |
| Archived weather forecasts cover only recent years | Shorter v2 train period; compare with/without weather on same test window |
| Hourly India data unavailable | India stays optional in v3 |
| Savings are small | Report them; analyze when and why savings vanish (low slack, high utilization, flat grids like France) |
| Average ≠ marginal emissions | E8, prominent limitation statement |
| Traces lack deadlines | Synthesized slack, sensitivity tests (E7) |
| Conformal coverage fails under shift | That is a finding; ACI + E2 |
| Oracle MILP too slow | Rolling 1-week windows, solver time limit, document gap |
| Colab disconnects mid-sweep | Resumable harness that skips finished configs |
| Scope creep | Freeze each version's scope (above) before starting it |

---

## 19. Definition of done (9/10 project)

- [ ] Clear research question with a stated, checked novelty claim
- [ ] ≥ 3 zones with different grid mixes, chronological evaluation
- [ ] Calibrated uncertainty shown with coverage plots, not just claimed
- [ ] Oracle upper bound and strong baselines in every comparison
- [ ] Ablations showing which parts matter
- [ ] Honest limitations, including average-vs-marginal caveat
- [ ] Reproducible open-source repo (one command per figure) and short paper-style report
- [ ] Only you appear as a contributor on GitHub
- [ ] You can explain every formula and design choice without notes

---

## 20. Glossary

- **Carbon intensity (CI):** grams of CO₂-equivalent emitted per kWh of electricity at a given time.
- **Horizon:** how far ahead a forecast predicts (1–48 h).
- **Issue time:** the moment a forecast is made; only data available then may be used.
- **Leakage:** accidentally using future information in training or features; makes results look better than reality.
- **Quantile / pinball loss:** loss function whose minimizer is a chosen quantile (e.g. the 90th percentile).
- **Coverage:** fraction of true values that fall inside predicted intervals.
- **Conformal prediction:** method that adjusts intervals using calibration data so coverage is guaranteed under exchangeability.
- **CQR:** conformal correction applied to quantile-regression intervals.
- **ACI:** online adjustment of the conformal level to keep coverage when the data distribution shifts.
- **Gray-box model:** a physics-based formula with a few parameters fitted to data.
- **LOMO / LOGO:** leave-one-model-out / leave-one-GPU-out cross-validation.
- **Slack:** extra time a job may wait beyond its run time before its deadline.
- **Oracle:** scheduler with perfect knowledge of the future; best achievable result.
- **MPC:** model predictive control — re-plan every step, execute only the first step.
- **MILP:** mixed-integer linear program — optimization with some yes/no variables.
- **CVaR:** conditional value at risk — average of the worst tail of outcomes.
- **Average vs marginal emissions:** average = emissions of the whole current mix; marginal = emissions of the extra generation caused by your extra demand.
- **Block bootstrap:** resampling contiguous blocks of time so confidence intervals respect time dependence.
