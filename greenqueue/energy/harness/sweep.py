"""Resumable measurement sweep: one JSON line per finished (config, repeat) (§9.1).

A disconnected Colab session is resumed by running the same sweep again: finished keys are skipped.
The workload runner is injected: ``runner(config) -> (callable step, flops_per_step)``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from greenqueue.energy.harness.power import (
    PowerLogger,
    Sampler,
    integrate_energy_kwh,
    measure_idle_w,
)


@dataclass(frozen=True)
class RunConfig:
    workload: str  # resnet18 | vit_b16 | bert_base | llm_infer
    model: str
    family: str  # cnn | vit | bert | llm
    batch: int
    seq: int = 0
    precision: str = "fp32"
    extra: tuple[tuple[str, int | str], ...] = ()  # e.g. (("prompt", 128), ("gen", 64))

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def default_grid() -> list[RunConfig]:
    """The sweep of implementation.md §9.1."""
    grid: list[RunConfig] = []
    for b in (32, 64, 128, 256):
        for prec in ("fp32", "fp16"):
            grid.append(RunConfig("resnet18", "resnet18", "cnn", b, 0, prec))
    for b in (16, 32, 64):
        grid.append(RunConfig("vit_b16", "vit_b16", "vit", b, 0, "fp16"))
    for b in (8, 16, 32):
        for s in (128, 256, 512):
            grid.append(RunConfig("bert_base", "bert_base", "bert", b, s, "fp16"))
    for model in ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B"):
        for b in (1, 4, 8, 16):
            for prompt in (128, 512):
                for gen in (64, 256):
                    grid.append(
                        RunConfig(
                            "llm_infer",
                            model,
                            "llm",
                            b,
                            prompt + gen,
                            "fp16",
                            (("gen", gen), ("prompt", prompt)),
                        )
                    )
    return grid


def quick_grid() -> list[RunConfig]:
    """A small grid that touches every family and every (prompt, gen) setting once.

    For short free-tier sessions: the first sweep stopped inside the LLM block, so some families
    and settings were never measured. Run this first, then the default grid to fill in.
    """
    g: list[RunConfig] = []
    for prec in ("fp32", "fp16"):
        for b in (32, 128):
            g.append(RunConfig("resnet18", "resnet18", "cnn", b, 0, prec))
    for b in (16, 64):
        g.append(RunConfig("vit_b16", "vit_b16", "vit", b, 0, "fp16"))
    for b in (8, 32):
        for sq in (128, 512):
            g.append(RunConfig("bert_base", "bert_base", "bert", b, sq, "fp16"))
    for b in (1, 4):
        for prompt in (128, 512):
            for gen in (64, 256):
                g.append(
                    RunConfig(
                        "llm_infer", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llm", b,
                        prompt + gen, "fp16", (("gen", gen), ("prompt", prompt)),
                    )
                )  # fmt: skip
    return g


def done_keys(runs_path: Path) -> set[tuple[str, int]]:
    if not runs_path.exists():
        return set()
    out = set()
    for line in runs_path.read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["config_key"], int(r["repeat"])))
    return out


def pending(
    grid: list[RunConfig], runs_path: Path, repeats: int
) -> Iterator[tuple[RunConfig, int]]:
    done = done_keys(runs_path)
    for cfg in grid:
        for rep in range(repeats):
            if (cfg.key(), rep) not in done:
                yield cfg, rep


def run_sweep(
    grid: list[RunConfig],
    runner: Callable[[RunConfig], tuple[Callable[[], None], float]],
    out_dir: Path,
    sampler: Sampler,
    gpu_info: dict[str, Any],
    *,
    repeats: int = 5,
    idle_s: float = 30.0,
    warmup_s: float = 10.0,
    min_run_s: float = 60.0,
    interval_s: float = 0.1,
) -> int:
    """Run every unfinished (config, repeat); returns the number of runs executed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    (out_dir / "traces").mkdir(exist_ok=True)
    executed = 0
    idle_cache: dict[str, float] = {}
    for cfg, rep in list(pending(grid, runs_path, repeats)):
        step, flops = runner(cfg)
        if cfg.key() not in idle_cache:  # idle measured once per config, before its repeats
            idle_cache[cfg.key()] = measure_idle_w(sampler, idle_s, interval_s)
        idle_w = idle_cache[cfg.key()]
        step()  # one un-timed step so lazy initialisation is not measured
        steps = 0
        t0 = time.perf_counter()
        with PowerLogger(sampler, interval_s) as log:
            while time.perf_counter() - t0 < min_run_s:
                step()
                steps += 1
        duration = time.perf_counter() - t0
        trace = log.trace()
        energy = integrate_energy_kwh(trace, idle_w, warmup_s)
        measured_s = float(trace.t_s[-1] - warmup_s)
        row = {
            "config_key": cfg.key(),
            "repeat": rep,
            **{k: v for k, v in asdict(cfg).items() if k != "extra"},
            **dict(cfg.extra),
            "steps": steps,
            "flops_per_step": flops,
            "duration_s": duration,
            "measured_s": measured_s,
            "energy_kwh_measured_window": energy,
            "step_time_s": duration / steps,
            "energy_per_step_kwh": energy / (steps * measured_s / duration),
            "mean_power_w": float(np.mean(trace.power_w)),
            "idle_w": idle_w,
            "mean_util_pct": float(np.mean(trace.util_pct)),
            "timestamp": time.time(),
            "source": "own",
            **gpu_info,
        }
        np.savez_compressed(
            out_dir / "traces" / f"{hashlib.md5(f'{cfg.key()}|{rep}'.encode()).hexdigest()}.npz",
            t=trace.t_s,
            p=trace.power_w,
            u=trace.util_pct,
        )
        with runs_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        executed += 1
    return executed
