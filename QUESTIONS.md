# QUESTIONS

## Q1: Python version and lockfile
Guide specifies Python 3.11 + uv lockfile. Local machine has 3.13 and 3.10, no uv.
Options: (a) install uv and Python 3.11, then `uv lock` (recommended, reproducible);
(b) stay on 3.13 with `pip freeze > requirements.lock`.

## Q2: Emission-factor values (decision made by default, please verify)
Work proceeded without the owner. IPCC AR5 medians were entered from memory of Annex III; entries
marked `assumption` in configs/emission_factors.yaml are not from that table. Verify before
reporting any ENTSO-E-based result. GB results use the NESO intensity directly and do not depend on this.

## Q3: NESO archived forecast has unknown vintage (decision made, please confirm)
`forecast` in the NESO API is a fixed short-lead forecast: its MAE is flat (~9 g/kWh) from horizon 12 to 48 h, so
it was not available at issue time for long horizons. It is kept only as a labelled reference (`*_vintage_unknown`)
and never used in scheduling experiments. The main model F3 uses lags and calendar only.

## Q4: Other defaults taken without the owner
- E1 fits once on the training period (no monthly refit); metrics are reported per month.
- Job power per GPU is an assumed lognormal (median 0.20 kW) because no GPU measurements exist.
- Alibaba trace weeks are resampled onto test weeks; 150 jobs per week; pool sized for 60% utilisation.
- Energy-estimator experiments (E3) use only synthetic data until real GPU measurements exist; see PROGRESS.md.
