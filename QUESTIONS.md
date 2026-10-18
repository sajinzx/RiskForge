# QUESTIONS

## Q1: Python version and lockfile
Guide specifies Python 3.11 + uv lockfile. Local machine has 3.13 and 3.10, no uv.
Options: (a) install uv and Python 3.11, then `uv lock` (recommended, reproducible);
(b) stay on 3.13 with `pip freeze > requirements.lock`.
