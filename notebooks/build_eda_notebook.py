"""Builds notebooks/01_eda.ipynb from the cells below (keeps the notebook reviewable in Git).

    python notebooks/build_eda_notebook.py
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Yellow Taxi vs. Uber/Lyft in New York City 2025: Exploratory Data Analysis (CRISP-DM phases 1 to 3)

**Business understanding.** Since app-based services entered the market, yellow taxis have lost most of their trips in New York City. Management of a mobility operator (or the regulator, the NYC Taxi & Limousine Commission) wants to know **where, when and at what price yellow taxis still compete with Uber and Lyft**, and which trips are likely to be served by a taxi rather than an app.

**Dataset.** NYC TLC Trip Record Data 2025, one Parquet file per month and source.
* `yellow_tripdata_2025-MM.parquet` Yellow Taxi trips (street hail, taxi meter)
* `fhvhv_tripdata_2025-MM.parquet` High Volume For-Hire Vehicle trips (Uber, Lyft)

**Analysis objective.** Compare the three services on demand, price, trip characteristics and tipping by zone, hour, weekday and month, and prepare a target variable and features for a model that predicts whether a trip is served by a yellow taxi.

**Unit of analysis.** One completed passenger trip (one row = one trip).

**Target variable.** `is_taxi` = 1 if the trip was served by a Yellow Taxi, 0 if by Uber or Lyft.

Data was downloaded with `src/download_tlc.py` and cleaned with the Prefect flow (`flows/etl_flow.py`).
""")

code("""
import sys
from pathlib import Path
sys.path.extend([str(Path.cwd().parent), str(Path.cwd().parent / "src")])

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dask.distributed import Client, LocalCluster

import config, context, pipeline

pd.set_option("display.float_format", "{:,.3f}".format)
plt.rcParams["figure.figsize"] = (10, 4.5)
COLORS = {"Yellow Taxi": "#F2B701", "Uber": "#222222", "Lyft": "#E0007A"}
SERVICES = ["Yellow Taxi", "Uber", "Lyft"]

cluster = LocalCluster(**config.DASK_CONFIGS["A_2workers_6GB"])
client = Client(cluster)
client
""")

md("## 1. Data volume and structure of the raw files")
code("""
raw = []
for f in sorted(config.RAW_DIR.glob("*_tripdata_*.parquet")):
    meta = pq.ParquetFile(f).metadata
    raw.append({"file": f.name, "source": f.name.split("_")[0], "size_mb": f.stat().st_size / 1e6,
                "rows": meta.num_rows, "columns": meta.num_columns})
raw = pd.DataFrame(raw)
display(raw.groupby("source").agg(files=("file", "count"), size_gb=("size_mb", lambda s: s.sum() / 1e3),
                                  rows=("rows", "sum"), columns=("columns", "max")))
print(f"TOTAL raw size: {raw.size_mb.sum() / 1e3:.2f} GB | rows: {raw.rows.sum():,}")
""")
md("The assignment requires more than 1.5 GB. One year of Yellow Taxi and Uber/Lyft trips is several times larger, which is why the processing is done with Dask instead of plain pandas.")

code("""
first = {s: sorted(config.RAW_DIR.glob(f"{s}_tripdata_*.parquet"))[0] for s in config.SOURCES}
for s, f in first.items():
    print(f"\\n=== Raw schema {s} ({f.name})")
    print(pq.read_schema(f).to_string(show_schema_metadata=False))
""")
md("""
The two sources have **different schemas** (e.g. `fare_amount` vs. `base_passenger_fare`, `tip_amount` vs. `tips`, only the app services report `driver_pay` and the request time). The pipeline therefore maps both to one **harmonised schema** (`pipeline.harmonise`) before any comparison.
""")

md("## 2. Harmonised data and data types")
code("""
ddf = pipeline.read_trips()
n_rows = len(ddf)
print(f"Rows after cleaning (trips): {n_rows:,} | columns: {len(ddf.columns)} | Dask partitions: {ddf.npartitions}")
desc = {
    "service": "Yellow Taxi, Uber or Lyft", "pickup_datetime": "Pickup time", "year": "Year", "month": "Month",
    "day": "Day of month", "day_of_week": "0 = Monday ... 6 = Sunday", "hour": "Pickup hour",
    "pu_location_id": "Pickup taxi zone (1..265)", "do_location_id": "Drop-off taxi zone",
    "trip_miles": "Trip distance (miles)", "duration_min": "Trip duration (minutes)",
    "fare": "Base fare without tip, taxes and surcharges (USD)", "tip": "Tip (USD)",
    "total_paid": "Total paid by the passenger (USD)", "cbd_fee": "Congestion relief zone toll (USD)",
    "driver_pay": "Driver pay, only Uber/Lyft (USD)", "wait_min": "Request to pickup, only Uber/Lyft (min)",
    "card_payment": "1 = paid by card / in app", "is_airport": "1 = pickup or drop-off at JFK, LGA or EWR",
    "is_taxi": "TARGET: 1 = Yellow Taxi", "source": "Raw source (yellow, fhvhv)",
}
pd.DataFrame({"dtype": ddf.dtypes.astype(str), "description": ddf.dtypes.index.map(desc)}).dropna()
""")

md("## 3. Data quality and cleaning")
code("""
q = pipeline.cleaning_summary()
rules = ["missing_key", "out_of_period", "invalid_duration", "invalid_distance", "invalid_fare", "implausible_speed"]
qs = q.groupby("source")[["rows_in", "rows_out", "negative_fare", "zero_distance", "unknown_zone", *rules]].sum()
share = qs[["negative_fare", "zero_distance", "unknown_zone", *rules]].div(qs["rows_in"], axis=0) * 100
share["removed_total"] = 100 - qs["rows_out"] / qs["rows_in"] * 100
display(qs[["rows_in", "rows_out"]])
share.T.rename_axis("share of raw rows (%)")
""")
md("""
Cleaning rules (thresholds in `config.py`), each row counted for the first rule it violates.
* **Out of period.** Pickup outside the month of the file (meter clock errors, late submissions).
* **Invalid duration.** Shorter than 1 minute or longer than 3 hours.
* **Invalid distance.** Zero distance (cancelled or not moving) or more than 100 miles.
* **Invalid fare.** Zero or **negative fares**. Negative fares are refunds and voids that exist only in the Yellow Taxi data.
* **Implausible speed.** Faster than 80 mph on average.

Trips with unknown zones (264, 265) are kept, but they appear as zone "Unknown" in the zone analysis.
""")

code("""
miss = (ddf[["fare", "tip", "total_paid", "driver_pay", "wait_min"]].isna()
        .groupby(ddf["service"]).mean().compute() * 100).round(1)
miss.rename_axis("missing (%)")
""")
md("""
Missing values are **structural**. `driver_pay` and `wait_min` do not exist for yellow taxis (the meter does not record a request time or the driver's share). They must not be imputed and are only compared between Uber and Lyft.
""")

md("""
### Distribution check: does cleaning change the data? (before vs. after)

Cleaning removes about 10 % of Yellow Taxi rows but only about 0.2 % of app rows, so we must verify that it does
not change **what the data represents**. The pipeline stores, for every source-month, the pickup-zone counts and
histograms of `fare`, `trip_miles` and `duration_min` **before and after** cleaning (`pipeline._accumulate_dist`,
saved in the `*_quality.json` files), so this comparison covers all trips, not a sample.

Criteria fixed before looking at the results:

1. **Geography within each service** (main test): the share of a service's trips per pickup borough may move by
   at most **0.5 percentage points**, and the zone-level total variation distance must stay below
   `config.MAX_ZONE_TVD` (= 0.02, also a data quality rule in RDS).
2. **Centre of the numeric distributions**: median shift at most `config.MAX_MEDIAN_SHIFT` (= 5 %). The far right
   tail (p99, max) is *expected* to shrink: removing it is the purpose of the cleaning rules.
3. **Taxi share inside a borough** (e.g. 5 % of trips in a borough were taxi before cleaning): reported, but not
   required to stay equal. Raw taxi data contains refunds and voids that app data does not, so the cleaned taxi
   share (15.3 % city-wide) is naturally lower than the raw share (about 16.7 %). What must not happen is a
   borough-specific collapse, which would mean a rule deletes trips of one area disproportionately.

If the files were cleaned before this check existed, `distribution_summary()` is empty: re-run the cleaning
(`python flows/etl_flow.py --full`) once to regenerate the `*_quality.json` files with distribution stats.
""")
code("""
dist = pipeline.distribution_summary()
assert not dist["zone"].empty, "No distribution stats found. Re-run: python flows/etl_flow.py --full"

zlookup = context.zone_lookup()
zone_b = dist["zone"].merge(zlookup[["location_id", "borough"]],
                             left_on="pu_location_id", right_on="location_id", how="left")
zone_b["borough"] = zone_b["borough"].fillna("Unknown")
zone_b.loc[zone_b.pu_location_id == -1, "borough"] = "Missing zone"

# Level 1: geography within each service (share of the service's own trips per borough)
bs = zone_b.groupby(["source", "service", "borough"])[["n_before", "n_after"]].sum().reset_index()
tot = bs.groupby("service")[["n_before", "n_after"]].transform("sum")
bs["share_before_%"] = bs.n_before / tot.n_before * 100
bs["share_after_%"] = bs.n_after / tot.n_after * 100
bs["delta_pp"] = bs["share_after_%"] - bs["share_before_%"]
bs["removed_%"] = (1 - bs.n_after / bs.n_before) * 100
bs["within_0.5pp"] = bs["delta_pp"].abs() <= 0.5
display(bs.sort_values(["service", "share_before_%"], ascending=[True, False])
          [["source", "service", "borough", "n_before", "n_after", "removed_%",
            "share_before_%", "share_after_%", "delta_pp", "within_0.5pp"]].round(2))

# Level 1b: same but per raw source, i.e. Yellow Taxi vs. all for-hire (contract) trips
src_b = zone_b.groupby(["source", "borough"])[["n_before", "n_after"]].sum().reset_index()
tot = src_b.groupby("source")[["n_before", "n_after"]].transform("sum")
src_b["share_before_%"] = src_b.n_before / tot.n_before * 100
src_b["share_after_%"] = src_b.n_after / tot.n_after * 100
src_b["delta_pp"] = src_b["share_after_%"] - src_b["share_before_%"]
display(src_b.round(2).rename(columns={"source": "source (yellow = taxi, fhvhv = contract)"}))
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
for ax, (src, label) in zip(axes, [("yellow", "Yellow Taxi"), ("fhvhv", "For-hire / contract (Uber, Lyft, ...)")]):
    d = src_b[src_b.source == src].set_index("borough").sort_values("share_before_%", ascending=False)
    d[["share_before_%", "share_after_%"]].plot(kind="bar", ax=ax, color=["#BBBBBB", "#4C72B0"], rot=30)
    ax.set(title=f"Share of {label} trips per pickup borough", xlabel="", ylabel="Share of trips (%)")
axes[0].legend(["before cleaning", "after cleaning"]); plt.tight_layout(); plt.show()

# Level 3: taxi share INSIDE each borough before vs. after (e.g. 5 % of a borough's trips were taxi)
bt = zone_b.assign(is_taxi=(zone_b.service == "Yellow Taxi").astype(int))
bt = bt.groupby(["borough", "is_taxi"])[["n_before", "n_after"]].sum().unstack(fill_value=0)
taxi_share = pd.DataFrame({
    "taxi_share_before_%": bt[("n_before", 1)] / bt["n_before"].sum(axis=1) * 100,
    "taxi_share_after_%": bt[("n_after", 1)] / bt["n_after"].sum(axis=1) * 100,
})
taxi_share["delta_pp"] = taxi_share["taxi_share_after_%"] - taxi_share["taxi_share_before_%"]
taxi_share.sort_values("taxi_share_before_%", ascending=False).round(2)
""")
code("""
# Level 2: numeric distributions per source (taxi vs. contract), before vs. after
hist_src = (dist["hist"].groupby(["source", "column", "bin_left", "bin_right"])[["n_before", "n_after"]]
            .sum().reset_index())
rows = []
for (src, col), d in hist_src.groupby(["source", "column"]):
    d = d.sort_values("bin_left")
    edges = pipeline.DIST_BINS[col]
    row = {"source": src, "column": col}
    for when in ("before", "after"):
        counts = d[f"n_{when}"].to_numpy()
        for q, name in [(0.25, "p25"), (0.5, "median"), (0.75, "p75"), (0.95, "p95")]:
            row[f"{name}_{when}"] = pipeline.hist_quantile(counts, edges, q)
    row["median_shift_%"] = (row["median_after"] / row["median_before"] - 1) * 100
    rows.append(row)
num_check = pd.DataFrame(rows)
sums_src = dist["sums"].groupby(["source", "column"])[["count_before", "sum_before",
                                                       "count_after", "sum_after"]].sum().reset_index()
num_check = num_check.merge(sums_src, on=["source", "column"])
num_check["mean_before"], num_check["mean_after"] = (num_check.sum_before / num_check.count_before,
                                                     num_check.sum_after / num_check.count_after)
display(num_check[["source", "column", "mean_before", "mean_after", "median_before", "median_after",
                   "median_shift_%", "p95_before", "p95_after"]].round(3))

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
for r, (src, label) in enumerate([("yellow", "Yellow Taxi"), ("fhvhv", "For-hire / contract")]):
    for c, col in enumerate(pipeline.DIST_COLUMNS):
        d = hist_src[(hist_src.source == src) & (hist_src.column == col)].sort_values("bin_left")
        d = d[np.isfinite(d.bin_left) & np.isfinite(d.bin_right)]   # open outer bins not plotted
        ax = axes[r, c]
        ax.plot(d.bin_left, d.n_before / d.n_before.sum(), color="#BBBBBB", label="before")
        ax.plot(d.bin_left, d.n_after / d.n_after.sum(), color="#4C72B0", label="after")
        ax.set(title=f"{label}: {col}", xlabel=col, ylabel="Share of trips")
axes[0, 0].legend(); plt.tight_layout(); plt.show()

bs.round(3).to_csv(config.OUTPUT_DIR / "distribution_before_after_borough.csv", index=False)
num_check.round(4).to_csv(config.OUTPUT_DIR / "distribution_before_after_numeric.csv", index=False)
print("saved outputs/distribution_before_after_borough.csv and _numeric.csv")
""")
md("""
**How to read this.** If `within_0.5pp` is true for every borough and the median shifts stay below 5 %, the
cleaning only removed invalid records (negative fares, zero distances, impossible speeds) without changing where
each service operates or what a typical trip looks like. A single borough failing the test would point to one
cleaning rule deleting that area disproportionately (for example airport zones) and would have to be investigated
before trusting any zone-level conclusion.
""")

md("## 4. Central tendency and dispersion by service")
code("""
num = ["fare", "trip_miles", "duration_min", "tip", "total_paid"]
rows = []
for s in SERVICES:
    sub = ddf[ddf.service == s][num]
    d, qn = dask.compute(sub.describe(), sub.quantile([0.5]))
    d.loc["median"] = qn.loc[0.5]
    rows.append(d.loc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]].T.assign(service=s))
stats = pd.concat(rows).reset_index(names="variable").set_index(["variable", "service"]).sort_index()
stats
""")
md("""
All money and distance variables are **right-skewed** (mean above median), because a small number of long airport trips pulls the mean up. For management reporting the median, or sums per trip computed from totals, are more robust than single averages.
""")

md("## 5. Distributions")
code("""
def hist_by_service(col, bins):
    def part(df):
        return pd.DataFrame({s: np.histogram(df.loc[df.service == s, col].clip(bins[0], bins[-1]), bins)[0]
                             for s in SERVICES})
    return ddf[["service", col]].map_partitions(part, meta={s: "i8" for s in SERVICES}).compute() \\
        .groupby(level=0).sum().set_index(pd.Index(bins[:-1]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, bins, label) in zip(axes, [("fare", np.arange(0, 101, 2), "Base fare (USD)"),
                                          ("trip_miles", np.arange(0, 25.5, 0.5), "Distance (miles)"),
                                          ("duration_min", np.arange(0, 91, 2), "Duration (minutes)")]):
    h = hist_by_service(col, bins)
    for s in SERVICES:
        ax.plot(h.index, h[s] / h[s].sum(), label=s, color=COLORS[s])
    ax.set(xlabel=label, ylabel="Share of trips", title=f"Distribution of {col}")
axes[0].legend(); plt.tight_layout(); plt.show()
""")

md("## 6. Market share over time (taxi vs. Uber/Lyft)")
code("""
monthly = ddf.groupby(["month", "service"]).agg(trips=("fare", "size"), revenue=("total_paid", "sum")).compute()
trips_m = monthly["trips"].unstack()[SERVICES]
share_m = trips_m.div(trips_m.sum(axis=1), axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
trips_m.plot(kind="bar", stacked=True, ax=ax1, color=[COLORS[s] for s in SERVICES], rot=0)
ax1.set(title="Trips per month", xlabel="Month 2025", ylabel="Trips")
share_m.plot(ax=ax2, color=[COLORS[s] for s in SERVICES], marker="o")
ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax2.set(title="Market share (trips)", xlabel="Month 2025"); plt.tight_layout(); plt.show()
display((share_m * 100).round(1))
""")

md("## 7. Patterns by hour, weekday and borough")
code("""
by_hour = ddf.groupby(["hour", "service"]).size().compute().unstack()[SERVICES]
by_dow = ddf.groupby(["day_of_week", "service"]).size().compute().unstack()[SERVICES]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
(by_hour / by_hour.sum()).plot(ax=ax1, color=[COLORS[s] for s in SERVICES])
ax1.set(title="Demand profile by pickup hour (share of each service's trips)", xlabel="Hour")
(by_hour["Yellow Taxi"] / by_hour.sum(axis=1)).plot(ax=ax2, color=COLORS["Yellow Taxi"], marker="o")
ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax2.set(title="Yellow Taxi market share by hour", xlabel="Hour"); plt.tight_layout(); plt.show()
by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
(by_dow.div(by_dow.sum(axis=1), axis=0) * 100).round(1).rename_axis("market share (%)")
""")
code("""
zones = context.zone_lookup()
by_zone = ddf.groupby(["pu_location_id", "service"]).size().compute().rename("trips").reset_index()
by_zone = by_zone.merge(zones, left_on="pu_location_id", right_on="location_id", how="left")
boro = by_zone.pivot_table(index="borough", columns="service", values="trips", aggfunc="sum").fillna(0)[SERVICES]
boro_share = boro.div(boro.sum(axis=1), axis=0)
ax = boro_share.sort_values("Yellow Taxi").plot(kind="barh", stacked=True, color=[COLORS[s] for s in SERVICES])
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.set(title="Market share by pickup borough", xlabel="Share of trips"); plt.tight_layout(); plt.show()
top = by_zone[by_zone.service == "Yellow Taxi"].nlargest(10, "trips")[["zone", "borough", "trips"]]
top.assign(share_of_taxi_trips=top.trips / by_zone[by_zone.service == "Yellow Taxi"].trips.sum())
""")
md("""
Yellow taxis are concentrated in **Manhattan and at the airports** (the "Yellow Zone" where street hails are allowed), while Uber and Lyft dominate the outer boroughs. Market share must therefore always be reported by zone, otherwise the city-wide figure hides where taxis are still competitive.
""")

md("## 8. Price comparison")
code("""
price = ddf.groupby(["service", "hour"]).agg(fare=("fare", "sum"), miles=("trip_miles", "sum")).compute()
fpm = (price.fare / price.miles).unstack(0)[SERVICES]
ax = fpm.plot(color=[COLORS[s] for s in SERVICES], marker=".")
ax.set(title="Base fare per mile by pickup hour", xlabel="Hour", ylabel="USD per mile"); plt.tight_layout(); plt.show()

dist_bins = [0, 1, 2, 3, 5, 10, 20, 100]
ddf_p = ddf[["service", "trip_miles", "fare"]].assign(
    dist_band=ddf.trip_miles.map_partitions(pd.cut, dist_bins, meta=("trip_miles", "category")).astype(str))
band = ddf_p.groupby(["dist_band", "service"])["fare"].mean().compute().unstack()[SERVICES]
band = band.reindex(pd.IntervalIndex.from_breaks(dist_bins).astype(str))
band.rename_axis("avg. base fare (USD) by distance band (miles)")
""")
md("""
The comparison uses the **base fare** (before tips, taxes and surcharges), because the surcharges differ by law between taxis and app services. Short trips are where taxis are most price competitive. App prices change with demand (surge pricing), which is visible in the hourly profile.
""")

md("## 9. Tips: why only card payments can be analysed")
code("""
yel = ddf[ddf.service == "Yellow Taxi"]
pay = yel.groupby("card_payment").agg(trips=("fare", "size"), tip_sum=("tip", "sum"), fare_sum=("fare", "sum")).compute()
pay["tip_rate"] = pay.tip_sum / pay.fare_sum
pay.index = pay.index.map({0: "cash / other", 1: "card"})
display(pay)
tips = ddf.groupby("service").agg(tipped=("tipped", "mean"), tip_sum=("card_tip", "sum"), fare_sum=("card_fare", "sum")).compute()
tips["tip_rate_card"] = tips.tip_sum / tips.fare_sum
tips.loc[SERVICES, ["tipped", "tip_rate_card"]].rename(columns={"tipped": "share of trips with tip"})
""")
md("""
In the Yellow Taxi data **cash tips are not recorded** (the meter only knows card tips), so tip analysis must be limited to card payments. For Uber and Lyft every trip is paid in the app, so all tips are recorded. Taxi passengers paying by card tip a much higher percentage than app passengers, partly because the taxi payment screen suggests 20 to 30 %.
""")

md("## 10. Target variable `is_taxi` and candidate features")
code("""
counts = ddf["is_taxi"].value_counts().compute()
display(counts.to_frame("trips").assign(share=lambda d: d.trips / d.trips.sum()))
print(f"Positive class share (Yellow Taxi): {counts.get(1, 0) / counts.sum():.1%}")
""")
md("""
The target is strongly **imbalanced**. A model must be evaluated with ROC-AUC, PR-AUC, precision and recall rather than accuracy, and needs class weights or resampling.

Candidate features known **at the moment a trip starts** (to avoid leakage, fare, tip and duration are excluded because they are a consequence of the service chosen):

| Feature | Source | Rationale |
|---|---|---|
| `hour`, `day_of_week`, `month` | pickup time | taxis are strongest in business hours on weekdays |
| `pu_borough`, `pu_service_zone`, `do_borough` | zone lookup (context data) | street hail is only allowed in the Yellow Zone and at airports |
| `trip_miles` | trip | taxis serve shorter trips |
| `is_airport` | zone | dedicated taxi ranks at JFK and LGA |
| `precip_mm`, `temp_mean_c` | weather (context data) | rain could push passengers to the nearest available car |
""")

md("## 11. Additional context data: weather")
code("""
weather = context.weather_daily(2025, 2025)
daily = ddf.groupby(["month", "day", "service"]).size().compute().rename("trips").reset_index()
daily["date"] = pd.to_datetime(dict(year=2025, month=daily.month, day=daily.day)).dt.date
d = daily.pivot_table(index="date", columns="service", values="trips").fillna(0)[SERVICES]
d["taxi_share"] = d["Yellow Taxi"] / d[SERVICES].sum(axis=1)
if weather is not None:
    d = d.join(weather.set_index("date"))
    d["rainy_day"] = d.precip_mm >= 5
    display(d.groupby("rainy_day")[["Yellow Taxi", "Uber", "Lyft", "taxi_share"]].mean())
    display(d[["Yellow Taxi", "Uber", "Lyft", "taxi_share", "precip_mm", "temp_mean_c"]].corr().loc[["precip_mm", "temp_mean_c"]].round(3))
else:
    print("Weather API not reachable. Continue without weather.")
""")

md("""
## 12. Model comparison and learning curve (CRISP-DM phases 4 and 5)

A single model with a single score proves only that prediction is *possible*. Here four models of increasing
complexity are trained on the **same sample, same features and same train/test split**, so their scores are
comparable:

| Model | Role |
|---|---|
| Dummy (stratified) | Random baseline: any real model must beat it clearly |
| Logistic regression | Linear baseline (borough / service-zone codes one-hot encoded, numeric features scaled) |
| Random forest | Non-linear ensemble, robust but slow |
| HistGradientBoosting | Gradient boosting, the strongest baseline of the previous version of this notebook |

Because only 15.3 % of trips are taxis, the models are ranked by **PR-AUC first** and ROC-AUC second (accuracy
would reward always predicting "app"). The winner then gets a **learning curve**: performance against training-set
size, with cross-validation, instead of one final number.
""")
code("""
import time

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

frac = min(1.0, 500_000 / n_rows)
cols = ["is_taxi", "hour", "day_of_week", "month", "day", "pu_location_id", "do_location_id", "trip_miles", "is_airport"]
s = ddf[cols].sample(frac=frac, random_state=1).compute()
z = zones.set_index("location_id")
for side, col in [("pu", "pu_location_id"), ("do", "do_location_id")]:
    s[f"{side}_borough"] = s[col].map(z.borough).fillna("Unknown").astype("category").cat.codes
s["pu_service_zone"] = s["pu_location_id"].map(z.service_zone).fillna("N/A").astype("category").cat.codes
if weather is not None:
    w = weather.assign(month=pd.to_datetime(weather.date).dt.month, day=pd.to_datetime(weather.date).dt.day)
    s = s.merge(w[["month", "day", "precip_mm", "temp_mean_c"]], on=["month", "day"], how="left")

features = [c for c in ["hour", "day_of_week", "month", "pu_borough", "do_borough", "pu_service_zone",
                        "trip_miles", "is_airport", "precip_mm", "temp_mean_c"] if c in s]
X_tr, X_te, y_tr, y_te = train_test_split(s[features], s["is_taxi"], test_size=0.3, stratify=s["is_taxi"], random_state=1)
print(f"Sample: {len(s):,} trips | train {len(X_tr):,} / test {len(X_te):,} | features: {features}")
print(f"Positive class in test: {y_te.mean():.1%} (a random model has PR-AUC = {y_te.mean():.3f})")
""")
code("""
cat_features = [c for c in ["pu_borough", "do_borough", "pu_service_zone"] if c in features]
num_features = [c for c in features if c not in cat_features]
pre_logreg = ColumnTransformer([
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_features),
])
estimators = {
    "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=1),
    "Logistic regression": Pipeline([("prep", pre_logreg),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1))]),
    "Random forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=1))]),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=1),
}

rows, fitted = [], {}
for name, est in estimators.items():
    t0 = time.perf_counter()
    fitted[name] = est.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    p = fitted[name].predict_proba(X_te)[:, 1]
    y_hat = (p >= 0.5).astype(int)
    rows.append({"model": name, "ROC-AUC": roc_auc_score(y_te, p), "PR-AUC": average_precision_score(y_te, p),
                 "precision": precision_score(y_te, y_hat, zero_division=0),
                 "recall": recall_score(y_te, y_hat, zero_division=0),
                 "F1": f1_score(y_te, y_hat, zero_division=0),
                 "balanced_accuracy": balanced_accuracy_score(y_te, y_hat), "fit_time_s": fit_s})
    print(f"{name:22s} fitted in {fit_s:6.1f}s")

comparison = (pd.DataFrame(rows).sort_values(["PR-AUC", "ROC-AUC"], ascending=False)
              .reset_index(drop=True))
display(comparison.round(3))
comparison.round(4).to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False)

best_name = comparison.loc[0, "model"]
model = fitted[best_name]          # winner, used by the learning curve and the importance plot below
print(f"Best model by PR-AUC: {best_name} "
      f"(ROC-AUC {comparison.loc[0, 'ROC-AUC']:.3f}, PR-AUC {comparison.loc[0, 'PR-AUC']:.3f})")

ax = comparison.set_index("model")[["ROC-AUC", "PR-AUC"]].sort_values("PR-AUC").plot(kind="barh")
ax.set(title="Model comparison on the held-out test set", xlabel="Score", xlim=(0, 1))
plt.tight_layout(); plt.show()
""")
md("""
### Learning curve of the winning model

The curve is computed **only on the training set** with 3-fold stratified cross-validation (the test set above
stays untouched), at 5 training sizes from 10 % to 100 %, for both ROC-AUC and PR-AUC. It answers the question a
single score cannot: *would more data help?*

* Validation curve still rising at 100 % → train the final model on more than the 500 k sample.
* Training curve far above validation → overfitting: the model memorises zones instead of learning patterns.
* Both curves close and flat → the features are saturated; better features, not more rows, are needed.

Note this takes several minutes: the winner is refitted 15 times per metric.
""")
code("""
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, learning_curve

cv = StratifiedKFold(3, shuffle=True, random_state=1)
train_sizes = np.linspace(0.1, 1.0, 5)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, (label, scoring) in zip(axes, [("ROC-AUC", "roc_auc"), ("PR-AUC", "average_precision")]):
    sizes, tr, te = learning_curve(clone(estimators[best_name]), X_tr, y_tr, train_sizes=train_sizes,
                                   cv=cv, scoring=scoring, n_jobs=-1, random_state=1, shuffle=True)
    for scores, name, color in [(tr, "train", "#BBBBBB"), (te, "cross-validation", "#4C72B0")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, label=name)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(title=f"{best_name}: {label}", xlabel="Training trips", ylabel=label)
axes[0].legend(); plt.tight_layout()
plt.savefig(config.OUTPUT_DIR / "learning_curve.png", dpi=150, bbox_inches="tight"); plt.show()
print("saved outputs/learning_curve.png")
""")
code("""
from sklearn.inspection import permutation_importance
imp = permutation_importance(model, X_te.iloc[:20_000], y_te.iloc[:20_000], scoring="roc_auc", n_repeats=3, random_state=1)
pd.Series(imp.importances_mean, index=features).sort_values().plot(kind="barh", title=f"Permutation importance, {best_name} (ROC-AUC drop)")
plt.tight_layout(); plt.show()
""")

md("""
## 13. Preliminary findings and need for additional information

Write the final numbers from the cells above into the report. The structure of the findings:

1. **Market structure.** Share of trips and revenue for Yellow Taxi, Uber and Lyft in 2025 (section 6).
2. **Where taxis compete.** Manhattan Yellow Zone and airports vs. outer boroughs (section 7).
3. **When taxis compete.** Hour and weekday profile (section 7).
4. **Price.** Fare per mile and by distance band (section 8).
5. **Tips.** Card tip rate taxi vs. app, with the cash tip limitation (section 9).
6. **Predictability.** The model comparison (Dummy, logistic regression, random forest, HistGradientBoosting on the same split) shows whether location and time are enough to tell taxi from app trips, and its learning curve shows whether a bigger training sample would still help (section 12).

**Additional context information needed.**
* **Taxi zone lookup** (TLC) is required, without it zone IDs cannot be interpreted. Already integrated.
* **Weather** (Open-Meteo, Central Park) to test the effect of rain. Already integrated.
* **Congestion pricing** (Manhattan toll since 5 January 2025, column `cbd_fee`) explains price differences in the CBD.
* Not available in the data and would be useful: surge multipliers of the apps, the number of active vehicles per service, and trips that were requested but not served.
""")

code("""
client.close(); cluster.close()
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "01_eda.ipynb"
nbf.write(nb, out)
print("written", out)
