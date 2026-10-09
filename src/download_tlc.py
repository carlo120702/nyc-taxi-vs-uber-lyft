"""Step 1 (Data acquisition): download the monthly TLC trip record files.

Usage (from the project folder, with the conda environment active):
    python src/download_tlc.py                         # yellow + fhvhv, all months START_YEAR..END_YEAR
    python src/download_tlc.py --source yellow         # only Yellow Taxi
    python src/download_tlc.py --year 2025 --month 3   # a single month

Files that already exist are skipped, so the script can be re-run safely
(for example when TLC publishes a new month). The taxi zone lookup and the
weather data (context) are downloaded as well.
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.extend([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)])
import config  # noqa: E402
import context  # noqa: E402


def raw_file_path(source: str, year: int, month: int) -> Path:
    return config.RAW_DIR / f"{source}_tripdata_{year}-{month:02d}.parquet"


def is_complete(path: Path) -> bool:
    """A Parquet file starts AND ends with the magic bytes PAR1. A download that was
    cut off by the server misses the end (footer) and cannot be read."""
    if path.suffix != ".parquet":
        return path.stat().st_size > 0
    with open(path, "rb") as fh:
        head = fh.read(4)
        fh.seek(-4, 2)
        return head == b"PAR1" and fh.read(4) == b"PAR1"


def download_file(url: str, target: Path, retries: int = 3) -> Path | None:
    if target.exists() and target.stat().st_size > 0:
        if is_complete(target):
            print(f"[skip] {target.name} already exists")
            return target
        print(f"[corrupt] {target.name} is incomplete, downloading it again")
    tmp = target.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {target.name} (attempt {attempt}) ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
                expected = int(resp.headers.get("Content-Length") or 0)
                while chunk := resp.read(4 * 1024 * 1024):
                    fh.write(chunk)
            if (expected and tmp.stat().st_size != expected) or not is_complete(tmp):
                raise IOError(f"incomplete download ({tmp.stat().st_size / 1e6:.0f} MB "
                              f"of {expected / 1e6:.0f} MB)")
            tmp.replace(target)
            print(f"[ok] {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except Exception as exc:  # network errors, 403/404 for months not yet published
            print(f"[error] {target.name}: {exc}")
            time.sleep(5 * attempt)
    if tmp.exists():
        tmp.unlink()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=config.SOURCES, help="only one source")
    parser.add_argument("--year", type=int, help="only one year")
    parser.add_argument("--month", type=int, help="only one month (needs --year)")
    args = parser.parse_args()

    sources = [args.source] if args.source else config.SOURCES
    years = [args.year] if args.year else range(config.START_YEAR, config.END_YEAR + 1)
    months = [args.month] if args.month else range(1, 13)

    failed = []
    for source in sources:
        for y in years:
            for m in months:
                url = config.TRIP_URL_TEMPLATE.format(source=source, year=y, month=m)
                if download_file(url, raw_file_path(source, y, m)) is None:
                    failed.append(f"{source} {y}-{m:02d}")

    context.zone_lookup()          # small CSV, cached in data/context
    context.weather_daily(min(years), max(years))

    total = sum(f.stat().st_size for f in config.RAW_DIR.glob("*.parquet")) / 1e9
    print(f"\nRaw data on disk: {total:.2f} GB (Parquet)")
    if failed:
        print("Failed (re-run the script to retry):", failed)


if __name__ == "__main__":
    main()
