# Data manifest

Raw and processed data are not committed (`data/raw/`, `data/processed/` are git-ignored). Only the
scripts that fetch them and this file are. **Licence and terms below were NOT verified against the
providers' current pages; the owner must read and confirm them (guide section 6).**

| Dataset | Source | Fetched | File | SHA-256 | Terms |
|---|---|---|---|---|---|
| GB carbon intensity (actual + NESO forecast), half-hourly 2018-01-01 to 2026-09-30 | NESO Carbon Intensity API, `https://api.carbonintensity.org.uk/intensity` | 2026-10-18 | `data/raw/uk_ci/intensity.parquet` | `6faff5917732c03303429ddda76752d4cf1ab01234771a2838851e07faf02a00` | OWNER TO VERIFY (public API, no key; rate limit not documented here, script pauses 0.3 s) |
| GB generation mix (percent by fuel) | same API, `/generation` | 2026-10-18 | `data/raw/uk_ci/generation.parquet` | `3f6cf15a7785c13a4b3466b7cf4561d6f4b161766ea6aee4234870d1110971c9` | OWNER TO VERIFY; 6.5% of half-hours missing; unused in v1 |
| Alibaba GPU cluster trace v2020 (PAI), task table | `https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz` (repo: `alibaba/clusterdata`, `cluster-trace-gpu-v2020`) | 2026-10-18 | `data/raw/traces/alibaba/pai_task_table.tar.gz` | `cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5` (matches the hash published in the trace README) | OWNER TO VERIFY; the authors ask users to cite Weng et al., NSDI 2022 |
| Alibaba task table header | `alibaba/clusterdata` repo | 2026-10-18 | `data/raw/traces/alibaba/pai_task_table.header` | `978bbaabfc8695874c605c01c144b2977f611ceca73aeb72189988cdfbfb0a9c` | as above |

Not yet obtained (blocked on the owner):

- ENTSO-E generation for DE, FR, PL, ES: needs an API token in `.env` (`data/scripts/download_entsoe.py`).
- GPU energy measurements: run `greenqueue/energy/harness/colab_runner.ipynb` on a GPU and place the output under
  `data/raw/energy_own/<gpu>/<date>/`.
- Open-Meteo archived forecasts, Philly / Helios / Azure traces, public energy-per-request tables.

Processed files derived from the above (not committed): `data/processed/gb_hourly.parquet`,
`data/processed/quality_report_GB.json`, `data/processed/alibaba_tasks.parquet`.
