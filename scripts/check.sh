#!/usr/bin/env bash
set -e
R=.venv/Scripts
$R/ruff format . -q
$R/ruff check .
$R/mypy greenqueue
$R/pytest
