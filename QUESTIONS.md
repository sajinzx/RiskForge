# QUESTIONS

## Q1: Python version and lockfile
Guide specifies Python 3.11 + uv lockfile. Local machine has 3.13 and 3.10, no uv.
Options: (a) install uv and Python 3.11, then `uv lock` (recommended, reproducible);
(b) stay on 3.13 with `pip freeze > requirements.lock`.

## Q2: Emission-factor values (decision made by default, please verify)
Work proceeded without the owner. IPCC AR5 medians were entered from memory of Annex III; entries
marked `assumption` in configs/emission_factors.yaml are not from that table. Verify before
reporting any ENTSO-E-based result. GB results use the NESO intensity directly and do not depend on this.
