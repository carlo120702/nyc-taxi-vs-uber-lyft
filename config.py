"""Central configuration for the NYC Taxi vs. Uber/Lyft Big Data project.

All paths, the analysis period, the Dask cluster configurations and the
database connection are defined here, so every script uses the same settings.
Secrets (database password) are read from a local `.env` file that is NOT
committed to GitHub.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE per line) without extra dependencies."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"            # original monthly TLC Parquet files (not in Git)
CLEAN_DIR = DATA_DIR / "clean"        # cleaned, harmonised trips, one file per source and month
CONTEXT_DIR = DATA_DIR / "context"    # zone lookup and weather (additional context data)
OUTPUT_DIR = ROOT / "outputs"         # benchmark results, figures, CSV exports

for _d in (RAW_DIR, CLEAN_DIR, CONTEXT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data source
# NYC Taxi & Limousine Commission (TLC) trip record data, one Parquet file per month.
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
TRIP_URL_TEMPLATE = TLC_BASE_URL + "/trip-data/{source}_tripdata_{year}-{month:02d}.parquet"
ZONE_LOOKUP_URL = TLC_BASE_URL + "/misc/taxi_zone_lookup.csv"

# yellow = Yellow Taxi, fhvhv = High Volume For-Hire Vehicles (Uber, Lyft)
SOURCES = ["yellow", "fhvhv"]
START_YEAR = int(os.environ.get("START_YEAR", 2025))
END_YEAR = int(os.environ.get("END_YEAR", 2025))

# HVFHS (High Volume For-Hire Services) license numbers -> company
HVFHS_COMPANIES = {"HV0002": "Juno", "HV0003": "Uber", "HV0004": "Via", "HV0005": "Lyft"}

# Weather (additional context): Open-Meteo historical archive, Central Park coordinates
WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=40.7812&longitude=-73.9665"
               "&start_date={start}&end_date={end}&timezone=America%2FNew_York"
               "&daily=temperature_2m_mean,precipitation_sum,snowfall_sum")

# ---------------------------------------------------------------- cleaning rules
# A trip is kept only if it passes all rules (see pipeline.clean_batch)
MIN_DURATION_MIN = 1        # shorter trips are cancellations or meter errors
MAX_DURATION_MIN = 180
MIN_MILES = 0.01
MAX_MILES = 100
MIN_FARE = 0.01
MAX_FARE = 500
MAX_SPEED_MPH = 80

# ---------------------------------------------------------------- Dask
# Machine used: Intel i7-1185G7 (4 cores / 8 logical), 32 GB RAM, ~16 GB free.
# Both configurations use 8 threads and 12 GB in total, so they are comparable.
DASK_CONFIGS = {
    "A_2workers_6GB": {"n_workers": 2, "threads_per_worker": 4, "memory_limit": "6GB"},
    "B_4workers_3GB": {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "3GB"},
}
DEFAULT_DASK_CONFIG = os.environ.get("DASK_CONFIG", "B_4workers_3GB")

# ---------------------------------------------------------------- database (AWS RDS PostgreSQL)
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "nyctaxi")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "prefer")
