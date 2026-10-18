"""Download archived Open-Meteo forecasts (previous_day2 vintage) for the zone points.

Usage: python data/scripts/download_weather.py --zone GB --start 2024-03-08 --end 2026-09-30
Points come from configs/zones.yaml. Writes data/raw/weather/<zone>_previous_day2.parquet
(hourly UTC, one column per variable and point).
"""

import argparse
import time
from datetime import date
from pathlib import Path

import requests
import yaml

from greenqueue.carbon.weather import combine_points, fetch_point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zone", default="GB")
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--zones-config", type=Path, default=Path("configs/zones.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/weather"))
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.zones_config.read_text(encoding="utf8"))
    variables = cfg["weather"]["variables"]
    session = requests.Session()

    def get_json(url: str, params: dict) -> dict:  # type: ignore[type-arg]
        time.sleep(args.pause)
        for attempt in range(5):
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt * 5)
                continue
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]
        raise RuntimeError("Open-Meteo kept failing")

    frames = {
        name: fetch_point(get_json, pt["lat"], pt["lon"], args.start, args.end, variables)
        for name, pt in cfg[args.zone]["points"].items()
    }
    wide = combine_points(frames)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.zone}_previous_day2.parquet"
    wide.to_parquet(path)
    print(path, wide.shape, "missing share:", float(wide.isna().mean().mean()))


if __name__ == "__main__":
    main()
