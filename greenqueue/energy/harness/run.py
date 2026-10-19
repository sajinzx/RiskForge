"""Entry point for the GPU machine: ``python -m greenqueue.energy.harness.run --out DIR``.

Prints the GPU, driver and CUDA version first (record them), then runs or resumes the sweep.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from greenqueue.energy.harness.power import NvmlSampler
from greenqueue.energy.harness.sweep import default_grid, quick_grid, run_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--min-run-s", type=float, default=60.0)
    ap.add_argument("--only", default=None, help="run only this workload name")
    ap.add_argument(
        "--quick", action="store_true", help="small grid, 3 repeats, shorter runs (short sessions)"
    )
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.idle_s, args.min_run_s = 3, 15.0, 30.0

    import torch

    sampler = NvmlSampler(0)
    info: dict[str, Any] = {
        **sampler.info(),
        "cuda": str(torch.version.cuda),
        "torch": torch.__version__,
    }
    print("GPU:", info)
    from greenqueue.energy.harness.workloads import make_runner

    base = quick_grid() if args.quick else default_grid()
    grid = [c for c in base if args.only in (None, c.workload)]
    out = args.out / info["gpu_name"].replace(" ", "_") / str(date.today())
    n = run_sweep(
        grid,
        lambda cfg: make_runner(cfg),
        out,
        sampler,
        info,
        repeats=args.repeats,
        idle_s=args.idle_s,
        min_run_s=args.min_run_s,
    )
    print(f"executed {n} runs into {out}")


if __name__ == "__main__":
    main()
