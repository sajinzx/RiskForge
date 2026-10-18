"""Download ENTSO-E actual generation per production type to data/raw/entsoe/<zone>.parquet.

Needs ENTSOE_API_KEY in the environment or .env. Usage:
  python data/scripts/download_entsoe.py --zones DE_LU FR PL ES --start 2018-01-01 --end 2026-09-30
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

from greenqueue.carbon.entsoe_api import download_zone


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", nargs="+", default=["DE_LU"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="inclusive day")
    parser.add_argument("--out", type=Path, default=Path("data/raw/entsoe"))
    args = parser.parse_args()
    _load_dotenv()
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("ENTSOE_API_KEY missing (set it in .env)")
    client = EntsoePandasClient(api_key=key)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    args.out.mkdir(parents=True, exist_ok=True)
    for zone in args.zones:
        df = download_zone(
            lambda z, a, b: client.query_generation(z, start=a, end=b, psr_type=None),
            zone,
            start,
            end,
        )
        df.columns = [" | ".join(c) if isinstance(c, tuple) else str(c) for c in df.columns]
        df.to_parquet(args.out / f"{zone}.parquet")
        print(zone, df.shape)


if __name__ == "__main__":
    main()
