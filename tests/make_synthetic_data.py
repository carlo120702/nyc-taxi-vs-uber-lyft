"""Generates small FAKE files in the exact TLC raw schemas (yellow and fhvhv), plus a
fake zone lookup and fake weather. They were only used to test the code without
downloading 6 GB, never for the analysis.

    set DATA_DIR=C:\\temp\\nyc_test        (Windows; so the real data folder is not touched)
    python tests/make_synthetic_data.py --months 1 2 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

rng = np.random.default_rng(42)
BOROUGHS = [("EWR", 1, 1), ("Queens", 2, 2), ("Bronx", 3, 3), ("Manhattan", 4, 4)]


def fake_zones() -> pd.DataFrame:
    rows = []
    for i in range(1, 266):
        if i == 1:
            b, sz = "EWR", "EWR"
        elif i in (132, 138):
            b, sz = "Queens", "Airports"
        elif i >= 264:
            b, sz = "Unknown", "N/A"
        elif i % 4 == 0 or 100 <= i <= 170:
            b, sz = "Manhattan", "Yellow Zone"
        else:
            b, sz = ["Brooklyn", "Queens", "Bronx", "Staten Island"][i % 4], "Boro Zone"
        rows.append({"LocationID": i, "Borough": b, "Zone": f"Zone {i}", "service_zone": sz})
    return pd.DataFrame(rows)


def _times(year, month, n):
    start = pd.Timestamp(year=year, month=month, day=1)
    days = pd.Period(start, "M").days_in_month
    hour_w = np.array([2, 1, 1, 1, 1, 2, 4, 6, 7, 6, 5, 5, 5, 5, 6, 7, 8, 8, 8, 7, 6, 5, 4, 3], float)
    hours = rng.choice(24, n, p=hour_w / hour_w.sum())
    t = start + pd.to_timedelta(rng.integers(0, days, n), "D") + pd.to_timedelta(hours, "h") \
        + pd.to_timedelta(rng.integers(0, 3600, n), "s")
    return pd.Series(t).astype("datetime64[us]")


def _zones(n, manhattan_bias):
    manhattan = np.array([z for z in range(1, 264) if z % 4 == 0 or 100 <= z <= 170])
    other = np.array([z for z in range(2, 264) if z not in set(manhattan)])
    pick = np.where(rng.random(n) < manhattan_bias, rng.choice(manhattan, n), rng.choice(other, n))
    airport = rng.random(n) < 0.04
    pick[airport] = rng.choice([132, 138, 1], airport.sum(), p=[0.6, 0.35, 0.05])
    pick[rng.random(n) < 0.01] = 264
    return pick.astype("int32")


def yellow(year, month, n):
    pu = _times(year, month, n)
    miles = np.round(rng.lognormal(0.6, 0.8, n), 2)
    dur = np.clip(miles * rng.normal(5, 1.5, n) + rng.normal(4, 2, n), 0.2, None)
    fare = np.round(3 + 2.8 * miles + 0.7 * dur, 2)
    pay = rng.choice([0, 1, 2, 3, 4], n, p=[0.05, 0.72, 0.18, 0.02, 0.03])
    tip = np.where(pay == 1, np.round(fare * rng.choice([0, .15, .2, .25, .3], n, p=[.15, .2, .35, .2, .1]), 2), 0)
    neg = rng.random(n) < 0.01
    fare[neg] *= -1
    miles[rng.random(n) < 0.015] = 0
    cbd = np.where(rng.random(n) < 0.35, 0.75, 0.0)
    airport_fee = np.where(np.isin(_zones(n, 0), [132, 138]) & (rng.random(n) < 0.5), 1.75, 0.0)
    total = np.round(fare + tip + 1 + 0.5 + 2.5 + cbd + airport_fee, 2)
    return pd.DataFrame({
        "VendorID": rng.choice([1, 2, 6, 7], n).astype("int32"),
        "tpep_pickup_datetime": pu,
        "tpep_dropoff_datetime": (pu + pd.to_timedelta(dur * 60, "s")).astype("datetime64[us]"),
        "passenger_count": rng.choice([1, 2, 3], n).astype("float64"),
        "trip_distance": miles, "RatecodeID": 1.0, "store_and_fwd_flag": "N",
        "PULocationID": _zones(n, 0.85), "DOLocationID": _zones(n, 0.7),
        "payment_type": pay.astype("int64"), "fare_amount": fare, "extra": 1.0, "mta_tax": 0.5,
        "tip_amount": tip, "tolls_amount": 0.0, "improvement_surcharge": 1.0, "total_amount": total,
        "congestion_surcharge": 2.5, "Airport_fee": airport_fee, "cbd_congestion_fee": cbd,
    })


def fhvhv(year, month, n):
    pu = _times(year, month, n)
    req = (pu - pd.to_timedelta(rng.gamma(2, 2.5, n) * 60, "s")).astype("datetime64[us]")
    miles = np.round(rng.lognormal(1.2, 0.8, n), 3)
    sec = np.clip(miles * rng.normal(240, 60, n) + rng.normal(300, 120, n), 10, None).astype("int64")
    fare = np.round(5 + 2.2 * miles + 0.6 * sec / 60 + rng.normal(0, 3, n), 2)
    lic = rng.choice(["HV0003", "HV0005"], n, p=[0.74, 0.26])
    cbd = np.where(rng.random(n) < 0.15, 1.5, 0.0)
    return pd.DataFrame({
        "hvfhs_license_num": lic, "dispatching_base_num": "B03404", "originating_base_num": "B03404",
        "request_datetime": req, "on_scene_datetime": req, "pickup_datetime": pu,
        "dropoff_datetime": (pu + pd.to_timedelta(sec, "s")).astype("datetime64[us]"),
        "PULocationID": _zones(n, 0.35), "DOLocationID": _zones(n, 0.35),
        "trip_miles": miles, "trip_time": sec, "base_passenger_fare": fare,
        "tolls": 0.0, "bcf": np.round(fare * 0.0275, 2), "sales_tax": np.round(fare * 0.08875, 2),
        "congestion_surcharge": np.where(rng.random(n) < 0.3, 2.75, 0.0), "airport_fee": 0.0,
        "tips": np.where(rng.random(n) < 0.18, np.round(fare * 0.15, 2), 0.0),
        "driver_pay": np.round(fare * rng.normal(0.75, 0.1, n), 2),
        "shared_request_flag": "N", "shared_match_flag": "N", "access_a_ride_flag": " ",
        "wav_request_flag": "N", "wav_match_flag": "N", "cbd_congestion_fee": cbd,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--months", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--yellow-rows", type=int, default=150_000)
    p.add_argument("--fhvhv-rows", type=int, default=600_000)
    a = p.parse_args()
    for m in a.months:
        yellow(a.year, m, a.yellow_rows).to_parquet(config.RAW_DIR / f"yellow_tripdata_{a.year}-{m:02d}.parquet", index=False)
        fhvhv(a.year, m, a.fhvhv_rows).to_parquet(config.RAW_DIR / f"fhvhv_tripdata_{a.year}-{m:02d}.parquet", index=False)
        print(f"fake month {a.year}-{m:02d} written to {config.RAW_DIR}")
    zone_file = config.CONTEXT_DIR / "taxi_zone_lookup.csv"
    if not zone_file.exists():
        fake_zones().to_csv(zone_file, index=False)
    days = pd.date_range(f"{a.year}-01-01", f"{a.year}-12-31")
    pd.DataFrame({"date": days.date, "temp_mean_c": np.round(12 - 12 * np.cos(2 * np.pi * days.dayofyear / 365), 1),
                  "precip_mm": np.round(rng.gamma(0.4, 6, len(days)), 1), "snow_cm": 0.0}) \
        .to_csv(config.CONTEXT_DIR / f"weather_{a.year}_{a.year}.csv", index=False)


if __name__ == "__main__":
    main()
