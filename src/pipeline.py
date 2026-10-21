"""Reusable transformation and analytics functions.

Used by the Prefect flow (flows/etl_flow.py), the Dask benchmark
(src/dask_benchmark.py) and the EDA notebook, so the logic exists only once.

Two raw sources with different schemas are harmonised into ONE common trip schema:
    yellow  Yellow Taxi trip records             (columns tpep_*, fare_amount, tip_amount, ...)
    fhvhv   High Volume For-Hire Vehicle records  (Uber, Lyft: base_passenger_fare, tips, driver_pay, ...)

Data layout after cleaning (one file per source and month):
    data/clean/yellow/2025/01.parquet   + 01_quality.json (cleaning statistics)
    data/clean/fhvhv/2025/01.parquet    + 01_quality.json
"""
import json
import sys
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.extend([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)])
import config  # noqa: E402

AIRPORT_ZONES = {1, 132, 138}          # Newark (EWR), JFK, LaGuardia
KNOWN_ZONES = (1, 263)                 # 264 and 265 = unknown / outside NYC

# Columns needed from each raw source (lower case; raw files mix upper and lower case)
RAW_COLUMNS = {
    "yellow": ["tpep_pickup_datetime", "tpep_dropoff_datetime", "pulocationid", "dolocationid",
               "trip_distance", "payment_type", "fare_amount", "tip_amount", "total_amount",
               "airport_fee", "cbd_congestion_fee"],
    "fhvhv": ["hvfhs_license_num", "request_datetime", "pickup_datetime", "dropoff_datetime",
              "pulocationid", "dolocationid", "trip_miles", "trip_time", "base_passenger_fare",
              "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee",
              "cbd_congestion_fee", "tips", "driver_pay"],
}

# Common (harmonised) schema of the cleaned trips
CLEAN_SCHEMA = pa.schema([
    ("service", pa.string()),            # Yellow Taxi | Uber | Lyft
    ("pickup_datetime", pa.timestamp("us")),
    ("year", pa.int16()), ("month", pa.int8()), ("day", pa.int8()),
    ("day_of_week", pa.int8()),          # 0 = Monday ... 6 = Sunday
    ("hour", pa.int8()),
    ("pu_location_id", pa.int16()), ("do_location_id", pa.int16()),
    ("trip_miles", pa.float32()), ("duration_min", pa.float32()),
    ("fare", pa.float32()),              # base fare without tips, taxes and surcharges
    ("tip", pa.float32()),
    ("total_paid", pa.float32()),        # everything the passenger paid
    ("cbd_fee", pa.float32()),           # Manhattan congestion relief zone toll (since 5 Jan 2025)
    ("driver_pay", pa.float32()),        # only Uber/Lyft
    ("wait_min", pa.float32()),          # only Uber/Lyft: request -> pickup
    ("card_payment", pa.int8()),         # Uber/Lyft always 1 (in-app payment)
    ("is_airport", pa.int8()),
])

RULES = ["missing_key", "out_of_period", "invalid_duration", "invalid_distance",
         "invalid_fare", "implausible_speed"]

# ---------------------------------------------------------------------
# Distribution check (before vs. after cleaning)
# Per source-month we store, in the *_quality.json next to the cleaned file:
#   zone_before / zone_after   trip counts per (service, pickup zone)
#   hist_before / hist_after   histogram counts per (service, column) on DIST_BINS
#   sums_before / sums_after   [count, sum] per (service, column) for exact means
# The raw rows themselves are never kept, so the distributions can still be
# compared over all ~290 M trips without storing a second copy of the data.
# Open outer bins (-inf / +inf) collect the invalid values that cleaning removes.
DIST_COLUMNS = ["fare", "trip_miles", "duration_min"]
DIST_BINS = {
    "fare": [-np.inf, 0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 500, np.inf],
    "trip_miles": [-np.inf, 0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100, np.inf],
    "duration_min": [-np.inf, 0, 1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, np.inf],
}


def _init_dist_stats(stats: dict) -> None:
    stats.setdefault("zone_before", {})
    stats.setdefault("zone_after", {})
    for when in ("before", "after"):
        stats.setdefault(f"hist_{when}", {})
        stats.setdefault(f"sums_{when}", {})


def _accumulate_dist(df: pd.DataFrame, stats: dict, when: str) -> None:
    """Add one (already harmonised) batch to the before/after distribution stats."""
    _init_dist_stats(stats)
    zones = stats[f"zone_{when}"]
    for (service, zone), n in df.groupby(["service", df["pu_location_id"].fillna(-1)]).size().items():
        key = f"{service}|{int(zone)}"
        zones[key] = zones.get(key, 0) + int(n)
    hists, sums = stats[f"hist_{when}"], stats[f"sums_{when}"]
    for col in DIST_COLUMNS:
        edges = np.asarray(DIST_BINS[col], dtype="float64")
        hcol = hists.setdefault(col, {})
        scol = sums.setdefault(col, {})
        for service, values in df.groupby("service")[col]:
            clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
            counts = np.histogram(clean, bins=edges)[0]
            hcol[service] = (np.asarray(hcol.get(service, np.zeros(len(edges) - 1)), dtype="int64")
                             + counts).tolist()
            prev = scol.get(service, [0, 0.0])
            scol[service] = [int(prev[0]) + int(clean.size), float(prev[1]) + float(clean.sum())]


def hist_quantile(counts, edges, q: float) -> float:
    """Quantile approximated from histogram counts (linear inside the bin)."""
    counts = np.asarray(counts, dtype="float64")
    edges = np.asarray(edges, dtype="float64")
    total = counts.sum()
    if total == 0:
        return float("nan")
    target = q * total
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, target, side="left"))
    if idx >= len(counts):
        idx = len(counts) - 1
    lo, hi = edges[idx], edges[idx + 1]
    if not np.isfinite(lo) or not np.isfinite(hi):   # open outer bin: report its finite edge
        return float(hi if np.isfinite(hi) else lo)
    before = cum[idx] - counts[idx]
    frac = 0.0 if counts[idx] == 0 else (target - before) / counts[idx]
    return float(lo + frac * (hi - lo))


# =====================================================================
# 1. Paths
# =====================================================================
def raw_path(source: str, year: int, month: int) -> Path:
    return config.RAW_DIR / f"{source}_tripdata_{year}-{month:02d}.parquet"


def clean_path(source: str, year: int, month: int) -> Path:
    return config.CLEAN_DIR / source / f"{year}" / f"{month:02d}.parquet"


def stats_path(source: str, year: int, month: int) -> Path:
    return clean_path(source, year, month).with_name(f"{month:02d}_quality.json")


def key_from_raw(path: Path) -> tuple[str, int, int]:
    # file name pattern: fhvhv_tripdata_2025-01.parquet
    source, _, period = Path(path).stem.split("_")
    year, month = period.split("-")
    return source, int(year), int(month)


def available_raw() -> list[tuple[str, int, int]]:
    return sorted(key_from_raw(p) for p in config.RAW_DIR.glob("*_tripdata_*.parquet")
                  if p.stem.split("_")[0] in config.SOURCES)


# =====================================================================
# 2. Harmonise and clean (one batch of rows)
# =====================================================================
def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").astype("float64") if col in df else pd.Series(np.nan, index=df.index)


def harmonise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a raw batch of either source to the common trip schema."""
    out = pd.DataFrame(index=df.index)
    if source == "yellow":
        out["service"] = "Yellow Taxi"
        pickup = pd.to_datetime(df["tpep_pickup_datetime"])
        dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
        out["trip_miles"] = _num(df, "trip_distance")
        out["fare"] = _num(df, "fare_amount")
        out["tip"] = _num(df, "tip_amount")
        out["total_paid"] = _num(df, "total_amount")
        out["driver_pay"] = np.nan
        out["wait_min"] = np.nan
        out["card_payment"] = (_num(df, "payment_type") == 1).astype("int8")
    else:
        out["service"] = df["hvfhs_license_num"].map(config.HVFHS_COMPANIES).fillna("Other FHV")
        pickup = pd.to_datetime(df["pickup_datetime"])
        dropoff = pd.to_datetime(df["dropoff_datetime"])
        out["trip_miles"] = _num(df, "trip_miles")
        out["fare"] = _num(df, "base_passenger_fare")
        out["tip"] = _num(df, "tips")
        fees = ["tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "cbd_congestion_fee"]
        out["total_paid"] = out["fare"] + out["tip"].fillna(0) + sum(_num(df, c).fillna(0) for c in fees)
        out["driver_pay"] = _num(df, "driver_pay")
        out["wait_min"] = (pickup - pd.to_datetime(df["request_datetime"])).dt.total_seconds() / 60
        out.loc[(out["wait_min"] < 0) | (out["wait_min"] > 120), "wait_min"] = np.nan
        out["card_payment"] = np.int8(1)

    out["pickup_datetime"] = pickup.astype("datetime64[us]")
    out["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60
    out["cbd_fee"] = _num(df, "cbd_congestion_fee").fillna(0)
    out["pu_location_id"] = _num(df, "pulocationid")
    out["do_location_id"] = _num(df, "dolocationid")
    return out


def clean_batch(df: pd.DataFrame, year: int, month: int, stats: dict) -> pd.DataFrame:
    """Apply the cleaning rules in a fixed order and count removed rows per rule
    (each row is counted for the FIRST rule it violates)."""
    speed = df["trip_miles"] / (df["duration_min"] / 60)
    masks = {
        "missing_key": df["pickup_datetime"].isna() | df["pu_location_id"].isna() | df["do_location_id"].isna(),
        "out_of_period": (df["pickup_datetime"].dt.year != year) | (df["pickup_datetime"].dt.month != month),
        "invalid_duration": ~df["duration_min"].between(config.MIN_DURATION_MIN, config.MAX_DURATION_MIN),
        "invalid_distance": ~df["trip_miles"].between(config.MIN_MILES, config.MAX_MILES),
        "invalid_fare": ~df["fare"].between(config.MIN_FARE, config.MAX_FARE),
        "implausible_speed": speed > config.MAX_SPEED_MPH,
    }
    # distribution of the raw batch (before cleaning) + informational counts
    _accumulate_dist(df, stats, "before")
    stats["rows_in"] += len(df)
    stats["negative_fare"] += int((df["fare"] < 0).sum())
    stats["zero_distance"] += int((df["trip_miles"] <= 0).sum())
    stats["unknown_zone"] += int((~df["pu_location_id"].between(*KNOWN_ZONES)).sum())

    keep = pd.Series(True, index=df.index)
    for rule in RULES:
        hit = masks[rule].fillna(True) & keep
        stats[rule] += int(hit.sum())
        keep &= ~hit
    df = df[keep].copy()
    _accumulate_dist(df, stats, "after")   # same distribution, after cleaning

    ts = df["pickup_datetime"].dt
    df["year"], df["month"], df["day"] = ts.year, ts.month, ts.day
    df["day_of_week"], df["hour"] = ts.dayofweek, ts.hour
    df["is_airport"] = (df["pu_location_id"].isin(AIRPORT_ZONES) | df["do_location_id"].isin(AIRPORT_ZONES)).astype("int8")
    stats["rows_out"] += len(df)

    dtypes = {f.name: f.type.to_pandas_dtype() for f in CLEAN_SCHEMA if f.name not in ("service", "pickup_datetime")}
    df = df.astype(dtypes)
    return df[[f.name for f in CLEAN_SCHEMA]]


# =====================================================================
# 3. Conversion raw month -> clean month (streaming, low memory)
# =====================================================================
def clean_month(source: str, year: int, month: int, overwrite: bool = False,
                batch_size: int = 1_000_000) -> Path:
    """Read one raw monthly file in batches (an Uber/Lyft month has ~20 million rows),
    harmonise, clean and write it as compressed Parquet. Cleaning statistics are
    saved next to it as JSON and later used for the data quality checks."""
    target = clean_path(source, year, month)
    if target.exists() and stats_path(source, year, month).exists() and not overwrite:
        return target

    pf = pq.ParquetFile(raw_path(source, year, month))
    name_map = {n.lower(): n for n in pf.schema_arrow.names}
    wanted = RAW_COLUMNS[source]
    missing = [c for c in wanted if c not in name_map]
    stats = {"source": source, "year": year, "month": month, "missing_columns": missing,
             **{k: 0 for k in ["rows_in", "rows_out", "negative_fare", "zero_distance", "unknown_zone", *RULES]}}
    _init_dist_stats(stats)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, CLEAN_SCHEMA, compression="snappy") as writer:
        for batch in pf.iter_batches(batch_size=batch_size, columns=[name_map[c] for c in wanted if c in name_map]):
            raw = batch.to_pandas()
            raw.columns = [c.lower() for c in raw.columns]
            df = clean_batch(harmonise(raw, source), year, month, stats)
            writer.write_table(pa.Table.from_pandas(df, schema=CLEAN_SCHEMA, preserve_index=False))
    tmp.replace(target)
    stats_path(source, year, month).write_text(json.dumps(stats, indent=2))
    return target


def clean_many_with_dask(keys: list[tuple[str, int, int]], overwrite: bool = False) -> list[Path]:
    """Clean several source-months in parallel on the active Dask cluster."""
    tasks = [dask.delayed(clean_month)(s, y, m, overwrite) for s, y, m in keys]
    return list(dask.compute(*tasks))


# =====================================================================
# 4. Data quality checks (per source and month)
# =====================================================================
def _dist_shift_metrics(s: dict) -> tuple[float, float]:
    """Largest zone-mix shift and largest median shift (before vs. after cleaning).

    Zone shift is the total variation distance between the pickup-zone shares of
    each service (0 = identical geography). Median shift is relative, per
    DIST_COLUMNS and service, from the stored histograms."""
    def shares(counts: dict) -> dict[str, dict[str, float]]:
        per_service: dict[str, dict[str, float]] = {}
        for key, n in counts.items():
            service, zone = key.rsplit("|", 1)
            per_service.setdefault(service, {})[zone] = float(n)
        return {svc: {z: n / max(sum(d.values()), 1) for z, n in d.items()}
                for svc, d in per_service.items()}

    before, after = shares(s.get("zone_before", {})), shares(s.get("zone_after", {}))
    tvd = 0.0
    for service in set(before) | set(after):
        zones = set(before.get(service, {})) | set(after.get(service, {}))
        tvd = max(tvd, 0.5 * sum(abs(before.get(service, {}).get(z, 0.0)
                                     - after.get(service, {}).get(z, 0.0)) for z in zones))

    median_shift = 0.0
    for col in DIST_COLUMNS:
        edges = DIST_BINS[col]
        services = set(s.get("hist_before", {}).get(col, {})) | set(s.get("hist_after", {}).get(col, {}))
        for service in services:
            med_b = hist_quantile(s["hist_before"][col].get(service, []), edges, 0.5)
            med_a = hist_quantile(s["hist_after"][col].get(service, []), edges, 0.5)
            if np.isfinite(med_b) and med_b != 0 and np.isfinite(med_a):
                median_shift = max(median_shift, abs(med_a - med_b) / abs(med_b))
    return float(tvd), float(median_shift)


def quality_checks(source: str, year: int, month: int) -> list[dict]:
    """Rule-based checks on the cleaning statistics. Each rule returns value, threshold and pass/fail."""
    s = json.loads(stats_path(source, year, month).read_text())
    n = max(s["rows_in"], 1)

    def rule(name, value, threshold, passed):
        return {"source": source, "year": year, "month": month, "check_name": name,
                "value": float(value), "threshold": float(threshold), "passed": bool(passed)}

    removed = 1 - s["rows_out"] / n
    checks = [
        rule("row_count_min", s["rows_in"], 100_000, s["rows_in"] >= 100_000),
        rule("missing_expected_columns", len(s["missing_columns"]), 0, len(s["missing_columns"]) == 0),
        rule("share_missing_key", s["missing_key"] / n, 0.01, s["missing_key"] / n <= 0.01),
        rule("share_out_of_period", s["out_of_period"] / n, 0.001, s["out_of_period"] / n <= 0.001),
        rule("share_negative_fare", s["negative_fare"] / n, 0.05, s["negative_fare"] / n <= 0.05),
        rule("share_zero_distance", s["zero_distance"] / n, 0.05, s["zero_distance"] / n <= 0.05),
        rule("share_unknown_zone", s["unknown_zone"] / n, 0.05, s["unknown_zone"] / n <= 0.05),
        rule("share_removed_total", removed, 0.10, removed <= 0.10),
    ]
    if "zone_before" in s:   # distribution stats exist (files cleaned with this version)
        tvd, median_shift = _dist_shift_metrics(s)
        checks += [
            rule("dist_zone_tvd_max", tvd, config.MAX_ZONE_TVD, tvd <= config.MAX_ZONE_TVD),
            rule("dist_median_shift_max", median_shift, config.MAX_MEDIAN_SHIFT,
                 median_shift <= config.MAX_MEDIAN_SHIFT),
        ]
    return checks


def cleaning_summary(keys: list[tuple[str, int, int]] | None = None) -> pd.DataFrame:
    """All cleaning statistics as one table (used in the EDA notebook)."""
    files = sorted(config.CLEAN_DIR.glob("*/*/*_quality.json"))
    rows = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame(rows)
    if keys is not None and len(df):
        df = df[[(s, y, m) in set(keys) for s, y, m in zip(df.source, df.year, df.month)]]
    return df


def distribution_summary(keys: list[tuple[str, int, int]] | None = None) -> dict[str, pd.DataFrame]:
    """Before/after-cleaning distributions aggregated over all cleaned source-months.

    Returns two tables:
      zone  one row per (source, service, pickup zone): n_before, n_after
      hist  one row per (source, service, column, bin): n_before, n_after,
            with bin_left / bin_right taken from DIST_BINS (outer bins are open)
      sums  one row per (source, service, column): count and sum before/after
            (exact means = sum / count)
    Files cleaned before this check existed have no distribution stats and are skipped.
    """
    files = sorted(config.CLEAN_DIR.glob("*/*/*_quality.json"))
    zone: dict[tuple, list] = {}
    hist: dict[tuple, list] = {}
    sums: dict[tuple, list] = {}
    for f in files:
        s = json.loads(f.read_text())
        if keys is not None and (s["source"], s["year"], s["month"]) not in set(keys):
            continue
        if "zone_before" not in s:
            continue
        for when, idx in (("before", 0), ("after", 1)):
            for key, n in s[f"zone_{when}"].items():
                service, zid = key.rsplit("|", 1)
                row = zone.setdefault((s["source"], service, int(zid)), [0, 0])
                row[idx] += int(n)
            for col in DIST_COLUMNS:
                edges = DIST_BINS[col]
                for service, counts in s.get(f"hist_{when}", {}).get(col, {}).items():
                    for b, n in enumerate(counts):
                        row = hist.setdefault((s["source"], service, col, b), [0, 0])
                        row[idx] += int(n)
                for service, (cnt, total) in s.get(f"sums_{when}", {}).get(col, {}).items():
                    row = sums.setdefault((s["source"], service, col), [0, 0.0, 0, 0.0])
                    row[2 * idx] += int(cnt)
                    row[2 * idx + 1] += float(total)
    zone_df = pd.DataFrame(
        [(src, svc, zid, v[0], v[1]) for (src, svc, zid), v in sorted(zone.items())],
        columns=["source", "service", "pu_location_id", "n_before", "n_after"])
    hist_rows = []
    for (src, svc, col, b), v in sorted(hist.items()):
        edges = DIST_BINS[col]
        hist_rows.append((src, svc, col, float(edges[b]), float(edges[b + 1]), v[0], v[1]))
    hist_df = pd.DataFrame(hist_rows, columns=["source", "service", "column",
                                               "bin_left", "bin_right", "n_before", "n_after"])
    sums_df = pd.DataFrame(
        [(src, svc, col, v[0], v[1], v[2], v[3]) for (src, svc, col), v in sorted(sums.items())],
        columns=["source", "service", "column", "count_before", "sum_before", "count_after", "sum_after"])
    return {"zone": zone_df, "hist": hist_df, "sums": sums_df}


# =====================================================================
# 5. Analytics with Dask (aggregated results that go to AWS)
# =====================================================================
def read_trips(keys: list[tuple[str, int, int]] | None = None, columns: list[str] | None = None) -> dd.DataFrame:
    """Lazy Dask DataFrame over all (or selected) cleaned monthly files of both sources."""
    if keys is None:
        files = sorted(config.CLEAN_DIR.glob("*/*/[0-9][0-9].parquet"))
    else:
        files = [clean_path(*k) for k in keys if clean_path(*k).exists()]
    if not files:
        raise FileNotFoundError("No cleaned Parquet files found. Run the cleaning step first.")
    # One Dask partition per Parquet row group (max. 1 million trips, ~150 MB in memory).
    # Without this, a whole Uber/Lyft month (~20 million trips, several GB) becomes ONE
    # partition and the worker is killed because it exceeds its memory limit.
    derived = columns is None
    if derived:   # pickup_datetime is not needed for the analytics, so it is not read
        columns = [f.name for f in CLEAN_SCHEMA if f.name != "pickup_datetime"]
    ddf = dd.read_parquet([str(f) for f in files], engine="pyarrow", columns=columns,
                          split_row_groups=True)
    if derived:
        # float64 for the sums: float32 loses precision when summing ~20 million trips per month
        for c in ("trip_miles", "duration_min", "fare", "tip", "total_paid", "cbd_fee", "driver_pay", "wait_min"):
            ddf[c] = ddf[c].astype("float64")
        ddf["is_taxi"] = (ddf["service"] == "Yellow Taxi").astype("int8")   # target variable
        ddf["source"] = ddf["is_taxi"].map({1: "yellow", 0: "fhvhv"}, meta=("source", "object"))
        ddf["tipped"] = (ddf["tip"] > 0).astype("int8")
        ddf["card_fare"] = ddf["fare"] * ddf["card_payment"]
        ddf["card_tip"] = ddf["tip"] * ddf["card_payment"]
        ddf["is_cbd"] = (ddf["cbd_fee"] > 0).astype("int8")
        ddf["has_wait"] = (~ddf["wait_min"].isna()).astype("int8")
    return ddf


SUMS = {"revenue": "total_paid", "fare_sum": "fare", "miles_sum": "trip_miles",
        "minutes_sum": "duration_min", "tip_sum": "tip"}


def _agg(ddf, keys, extra=()):
    spec = {"trips": ("fare", "size"), **{k: (v, "sum") for k, v in SUMS.items()}}
    for name in extra:
        spec[name] = EXTRA_SUMS[name]
    return ddf.groupby(keys).agg(split_out=1, **spec)


EXTRA_SUMS = {
    "tipped_trips": ("tipped", "sum"), "card_trips": ("card_payment", "sum"),
    "card_fare_sum": ("card_fare", "sum"), "card_tip_sum": ("card_tip", "sum"),
    "cbd_trips": ("is_cbd", "sum"), "cbd_fee_sum": ("cbd_fee", "sum"),
    "airport_trips": ("is_airport", "sum"), "driver_pay_sum": ("driver_pay", "sum"),
    "wait_min_sum": ("wait_min", "sum"), "wait_trips": ("has_wait", "sum"),
}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df["trips"] = df["trips"].astype("int64")
    df["avg_fare"] = df["fare_sum"] / df["trips"]
    df["avg_miles"] = df["miles_sum"] / df["trips"]
    df["avg_duration_min"] = df["minutes_sum"] / df["trips"]
    df["fare_per_mile"] = df["fare_sum"] / df["miles_sum"].where(df["miles_sum"] > 0)
    df["avg_speed_mph"] = df["miles_sum"] / (df["minutes_sum"] / 60).where(df["minutes_sum"] > 0)
    return df


def build_aggregates(ddf: dd.DataFrame, zones: pd.DataFrame | None = None,
                     weather: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Compute all result tables in ONE Dask pass (shared reading of the data)."""
    base = ["year", "month", "source", "service"]
    service, zone, hour, daily = dask.compute(
        _agg(ddf, base, extra=EXTRA_SUMS),
        _agg(ddf, base + ["pu_location_id"], extra=["cbd_trips", "airport_trips"]),
        _agg(ddf, base + ["day_of_week", "hour"], extra=["wait_min_sum", "wait_trips"]),
        _agg(ddf, base + ["day"], extra=["cbd_trips"]),
    )

    service = _finish(service)
    service["tipped_share"] = service["tipped_trips"] / service["trips"]
    service["tip_rate_card"] = service["card_tip_sum"] / service["card_fare_sum"].where(service["card_fare_sum"] > 0)
    service["cbd_share"] = service["cbd_trips"] / service["trips"]
    service["airport_share"] = service["airport_trips"] / service["trips"]
    service["avg_wait_min"] = service["wait_min_sum"] / service["wait_trips"].where(service["wait_trips"] > 0)
    is_app = service["source"] == "fhvhv"
    service["driver_pay_share"] = (service["driver_pay_sum"] / service["fare_sum"]).where(is_app)

    zone = _finish(zone)
    if zones is not None:
        zone = zone.merge(zones.rename(columns={"location_id": "pu_location_id"}), on="pu_location_id", how="left")
    for c in ("borough", "zone", "service_zone"):
        if c not in zone:
            zone[c] = "Unknown"
        zone[c] = zone[c].fillna("Unknown")

    hour = _finish(hour)
    hour["avg_wait_min"] = hour["wait_min_sum"] / hour["wait_trips"].where(hour["wait_trips"] > 0)

    daily = _finish(daily)
    daily["date"] = pd.to_datetime(dict(year=daily.year, month=daily.month, day=daily.day)).dt.date
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    for c in ("temp_mean_c", "precip_mm", "snow_cm"):
        daily[c] = np.nan
    if weather is not None:
        daily = daily.drop(columns=["temp_mean_c", "precip_mm", "snow_cm"]).merge(weather, on="date", how="left")

    return {"service_month_kpi": service, "zone_month_kpi": zone,
            "hour_weekday_kpi": hour, "daily_kpi": daily}
