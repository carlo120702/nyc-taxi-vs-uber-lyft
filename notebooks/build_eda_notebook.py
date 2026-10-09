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

md("## 12. Baseline model (feasibility check for the next CRISP-DM phase)")
code("""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=1).fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
print(f"Sample: {len(s):,} trips | features: {features}")
print(f"ROC-AUC: {roc_auc_score(y_te, p):.3f} | PR-AUC: {average_precision_score(y_te, p):.3f} "
      f"(random baseline PR-AUC = {y_te.mean():.3f})")
""")
code("""
from sklearn.inspection import permutation_importance
imp = permutation_importance(model, X_te.iloc[:20_000], y_te.iloc[:20_000], scoring="roc_auc", n_repeats=3, random_state=1)
pd.Series(imp.importances_mean, index=features).sort_values().plot(kind="barh", title="Permutation importance (ROC-AUC drop)")
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
6. **Predictability.** The baseline model shows whether location and time are enough to tell taxi from app trips (section 12).

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
