"""Download GB carbon intensity (with NESO's forecast) and generation mix to data/raw/uk_ci/.

Usage: python data/scripts/download_uk_ci.py --start 2018-01-01 --end 2026-09-30
``--end`` is inclusive (the whole of that day is fetched).
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from greenqueue.carbon.uk_api import download_range, row_count_report, save_parquet


def _day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_day)
    parser.add_argument("--end", required=True, type=_day)
    parser.add_argument("--out", type=Path, default=Path("data/raw/uk_ci"))
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    end = args.end + timedelta(days=1)
    intensity, mix = download_range(args.start, end, pause_s=args.pause)
    save_parquet(intensity, args.out / "intensity.parquet")
    save_parquet(mix, args.out / "generation.parquet")

    report = {
        "intensity": row_count_report(intensity, args.start, end),
        "generation": row_count_report(mix, args.start, end),
        "actual_nan": int(intensity["actual"].isna().sum()),
    }
    (args.out / "download_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
