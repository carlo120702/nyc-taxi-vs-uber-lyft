"""Additional context data (CRISP-DM: "is additional information needed?").

1. Taxi zone lookup (TLC): translates the numeric LocationID of every trip into
   borough, zone name and service zone (Yellow Zone, Boro Zone, Airports, EWR).
2. Daily weather for Central Park (Open-Meteo historical archive, free, no API key):
   mean temperature, precipitation and snowfall, to test whether bad weather
   shifts demand between taxis and Uber/Lyft.

Both files are cached in data/context, so they are only downloaded once.
"""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ZONE_FILE = config.CONTEXT_DIR / "taxi_zone_lookup.csv"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def zone_lookup() -> pd.DataFrame:
    """LocationID -> borough, zone, service_zone (265 rows)."""
    if not ZONE_FILE.exists():
        ZONE_FILE.write_bytes(_get(config.ZONE_LOOKUP_URL))
    z = pd.read_csv(ZONE_FILE)
    z.columns = [c.lower() for c in z.columns]
    z = z.rename(columns={"locationid": "location_id"})
    for c in ("borough", "zone", "service_zone"):
        z[c] = z[c].fillna("Unknown").astype(str)
    return z[["location_id", "borough", "zone", "service_zone"]]


def weather_daily(start_year: int, end_year: int) -> pd.DataFrame | None:
    """Daily weather in Central Park. Returns None if the API is not reachable
    (the pipeline then simply continues without weather columns)."""
    target = config.CONTEXT_DIR / f"weather_{start_year}_{end_year}.csv"
    if not target.exists():
        end = min(pd.Timestamp(f"{end_year}-12-31"), pd.Timestamp.today().normalize() - pd.Timedelta(days=5))
        url = config.WEATHER_URL.format(start=f"{start_year}-01-01", end=end.strftime("%Y-%m-%d"))
        try:
            daily = json.loads(_get(url))["daily"]
        except Exception as exc:
            print(f"[warning] weather data not available: {exc}")
            return None
        pd.DataFrame(daily).rename(columns={
            "time": "date", "temperature_2m_mean": "temp_mean_c",
            "precipitation_sum": "precip_mm", "snowfall_sum": "snow_cm"}).to_csv(target, index=False)
    w = pd.read_csv(target, parse_dates=["date"])
    w["date"] = w["date"].dt.date
    return w
