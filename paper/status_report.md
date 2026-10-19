# GreenQueue status report (2026-10-19)

Owner: Sajin MK. Source of tasks: the roadmap in `implementation.md` section 14.

## How the percentages are computed

Each roadmap task has a weight of 1 (41 core tasks: 28 in v1, 10 in v2, 3 in v3; v4 and the optional v3 tasks
are excluded). Credit is 1 when finished and evidenced, 0.25-0.75 when partly done or waiting for the owner,
0 when not started. This is a judgement, not a measurement. Tasks needing the owner or supervisor are counted
separately, because the code cannot move them.

## Summary

| Scope | Done | Pending |
|---|---|---|
| v1 (28 tasks) | 70% | 30% |
| v2 (10 tasks) | 76% | 24% |
| v3 core (3 tasks) | 58% | 42% |
| **All core tasks (41)** | **70%** | **30%** |
| Tasks Claude Code can do alone (26) | 96% | 4% |
| Tasks needing the owner or a reviewer (15) | 27% | 73% |

So most of what remains needs you, your data access, or a reviewer.

## Done (with evidence)

- Carbon data pipeline for Great Britain (hourly intensity, 2018 to 2026-09), quality report, leakage tests.
- Forecasters F0 to F4, conformal methods (split, CQR, ACI), E1 / E2 / EW results. Raw intervals under-cover
  (74.6% at a nominal 90%); split conformal 90.7%; ACI 90.0%; weather lowers MAE from 37.2 to 24.5 but hurts 1-6 h.
- Simulator with exact accounting, schedulers S0 to S4, oracle MILP, E4 / E5 / E6 / E7 / E8 results.
- E0: perfect-foresight shifting potential (task T1.4); a 1 h job with a 24 h window saves 32.7% on average,
  a 24 h job 12.6% (GB, 2024-2026). Not compared with published numbers, which I have not verified.
- Energy harness ran on a real Tesla T4 (114 runs, 23 configurations); energy models fitted (E3).
- README, progress log, dashboard, quick GPU grid, one-command `reproduce v1|v2|v3`.
- Quality gates: ruff and mypy clean; test suite passing at the last full run (before the dashboard and E0 were added).

## Pending

**Blocked on the owner**

| Item | Why it matters | Roadmap task |
|---|---|---|
| ENTSO-E token (email sent, no reply yet) | Zones DE, FR, PL, ES, so E9 cross-zone transfer and the v1 DE zone | T0.4, T1.3, T7.2, E9 |
| More GPU measurements, ideally a second GPU type | The T4 sweep was partial: energy models are 40-100% off and intervals are infinite; leave-one-GPU-out is undefined | T3.2, T8.1 |
| Verify emission factors, GPU peak TFLOPS, data licences | All carbon numbers depend on the factors | T0.4 |
| Literature review and citations | Related work is preliminary; Li et al. 2024 already uses conformal prediction for carbon forecasts | T0.3 |
| Read and interpret results, write the v1 / v2 / v3 reports | Required deliverables | T2.5, T5.4, T6.2, T10.2, T11.6 |
| Supervisor reviews | Novelty, report | T1.5, T6.3 |
| Dependency lockfile (needs Python 3.11 and uv) | Reproducibility | Q1 |

**Claude Code work still open**

- E9 cross-zone transfer and DE zone runs (waiting for the token).
- Philly trace loader (needs the trace download), part of T5.1.
- Rerun scheduling with a usable energy estimator once more GPU data exists.
- Full test suite rerun after the latest changes.

**Optional, not started:** India analysis, LLM inference trace study, decision-latency benchmark, v4 theory.

## Risks and honest caveats

- Scheduling results use a synthetic duration and energy estimator; GPU power is assumed.
- The oracle bound is a time-limited MILP incumbent at 60% load (25 blocks hit the limit).
- Marginal-emission results are proxies, not measured marginal data.
- The MPC risk mode was chosen on the calibration year after one diagnostic looked at test weeks (disclosed).
- Only one zone and one GPU type have real data.
