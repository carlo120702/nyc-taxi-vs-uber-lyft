# Yellow Taxi vs. Uber/Lyft in New York City 2025: Big Data Group Project

Analysis of the NYC Taxi & Limousine Commission (TLC) trip records for 2025, comparing **Yellow Taxi** with the app-based services **Uber and Lyft** (High Volume For-Hire Vehicles), following the CRISP-DM methodology (Cross-Industry Standard Process for Data Mining).

**Business question.** Where, when and at what price can yellow taxis still compete with Uber and Lyft, and which trips are likely to be served by a taxi?

| Requirement | Where |
|---|---|
| a. Dataset, objective, unit of analysis | `notebooks/01_eda.ipynb` (introduction) |
| b. EDA, target variable, features | `notebooks/01_eda.ipynb` (sections 1 to 10) |
| c. Preliminary analysis, context data | `notebooks/01_eda.ipynb` (sections 11 to 13), `src/context.py` |
| d. Recurring, scalable ETL with Prefect | `flows/etl_flow.py` |
| e. Dask with two worker / memory configurations | `src/dask_benchmark.py`, `config.py` |
| f. Database in AWS (results only) | `src/db.py` (AWS RDS PostgreSQL, SQLAlchemy ORM) |
| g. Management dashboard | `grafana/` |
| h. Conclusions, i to k. Report | `report/NYC_Taxi_vs_Uber_Lyft_Report.docx` |
| l. Code repository | this repository |

## CRISP-DM mapping

| Phase | Where in this repository |
|---|---|
| 1. Business understanding | `notebooks/01_eda.ipynb` (introduction): business question, unit of analysis, target `is_taxi` |
| 2. Data understanding | `notebooks/01_eda.ipynb` (sections 1 to 9): volume, schemas, data quality, distributions |
| 3. Data preparation | `src/pipeline.py` (`harmonise`, `clean_batch`), `src/context.py` (zones, weather). Distribution before vs. after cleaning (per service and pickup borough) is checked in section 3 of the notebook and by two quality rules (`dist_zone_tvd_max`, `dist_median_shift_max`) |
| 4. Modeling | `notebooks/01_eda.ipynb` (section 12): four models compared on the same sample and split (Dummy, logistic regression, random forest, HistGradientBoosting), table in `outputs/model_comparison.csv` |
| 5. Evaluation | ROC-AUC and PR-AUC instead of accuracy because Yellow Taxi is only 15.3 % of trips, plus a learning curve (ROC-AUC and PR-AUC vs. training size, 3-fold CV) of the winning model in `outputs/learning_curve.png` — not a single score |
| 6. Deployment | `flows/etl_flow.py` (Prefect), `src/db.py` (AWS RDS), `grafana/` (dashboard) |

## Tools used

| Tool | Role in the project |
|---|---|
| VS Code | Editor and notebook runner |
| Parquet / PyArrow | Columnar file format for raw and cleaned trips, read in batches of 1 million rows |
| Dask | Parallel processing of 287 M trips (two cluster configurations benchmarked) |
| Prefect | Recurring, incremental ETL flow with retries and run log |
| AWS RDS (PostgreSQL) | Stores only the aggregated KPI tables, quality checks and run log |
| SQLAlchemy ORM | Python classes for the tables, idempotent load |
| Grafana (Docker) | Management dashboard on top of the RDS tables |
| scikit-learn | Model comparison (4 classifiers) and learning curve to predict `is_taxi` |

## Data

| Source | Files | Content |
|---|---|---|
| `yellow_tripdata_2025-MM.parquet` | 12 | Yellow Taxi trips (taxi meter) |
| `fhvhv_tripdata_2025-MM.parquet` | 12 | Uber and Lyft trips (column `hvfhs_license_num`: HV0003 Uber, HV0005 Lyft) |
| `taxi_zone_lookup.csv` | 1 | LocationID to borough, zone and service zone (context) |
| Open-Meteo archive | 1 | Daily temperature, rain and snow in Central Park (context) |

Together about 6 GB and roughly 250 to 300 million trips, well above the 1.5 GB requirement. Unit of analysis is one completed trip. Target variable is `is_taxi` (1 = Yellow Taxi, 0 = Uber or Lyft).

## Architecture

```
TLC website ──download_tlc.py──> data/raw/*.parquet  (≈ 6 GB, local only)
                                        │
                     Prefect flow "nyc-taxi-vs-uber-lyft-etl"
                                        │
  ┌────────────┬──────────────┬─────────┴──────────┬───────────────┬────────────────┐
  discover     context        clean (Dask)         validate        analyze (Dask)   load (ORM)
  new files    zones +        harmonise yellow +   10 quality      4 KPI tables     AWS RDS PostgreSQL
               weather        fhvhv schemas,       rules per                              │
                              remove outliers      file                               Grafana
```

Only aggregated KPI tables, quality results and the run log are stored in AWS. The raw trip records never leave the local machine.

### Result tables in AWS RDS

| Table | Grain | Content |
|---|---|---|
| `service_month_kpi` | year, month, service | trips, passenger spending, fares, distance, duration, tips, congestion toll, airport trips, wait time, driver pay |
| `zone_month_kpi` | year, month, pickup zone, service | trips, spending, fares, with borough and zone name |
| `hour_weekday_kpi` | year, month, weekday, hour, service | demand, fare per mile, speed, wait time |
| `daily_kpi` | date, service | trips and spending with temperature and precipitation |
| `quality_check` | run, source, month, rule | result of each data quality rule |
| `etl_run_log` | run | start, end, status, Dask configuration, files processed |

All tables store **sums** (not only averages), so Grafana can correctly re-aggregate any time range.

### Dask configurations (machine: i7-1185G7, 4 cores / 8 threads, 32 GB RAM)

| Config | Workers | Threads per worker | Memory per worker | Total |
|---|---|---|---|---|
| A_2workers_6GB | 2 | 4 | 6 GB | 8 threads, 12 GB |
| B_4workers_3GB | 4 | 2 | 3 GB | 8 threads, 12 GB |

Benchmark result over 287.1 M trips (median of 2 runs): A took 286 s, B 441 s. Configuration A is therefore the default.

## Setup (Windows, Anaconda Prompt)

Run each command separately, one after the other.

1. Activate the course environment
   ```
   conda activate usab_clase2
   ```
2. Go to the project folder
   ```
   cd "path\to\Group_Project_NYC_Taxi"
   ```
3. Install the packages
   ```
   pip install -r requirements.txt
   ```
4. Create your `.env` file by copying `.env.example` and filling in the RDS endpoint and password.

## AWS RDS (Learner Lab)

Create a PostgreSQL instance as in Assignment 1 (Free tier template, `db.t3.micro`, public access *Yes*, security group allows port 5432). Under *Additional configuration* set **Initial database name** to `nyctaxi`. Copy the endpoint into `.env`.

## Run

1. Download the data (24 files, about 6 GB, takes a while)
   ```
   python src/download_tlc.py
   ```
2. Start the Prefect server in a **second** terminal (activate the environment there too) and keep it open
   ```
   prefect server start
   ```
   In the first terminal, point Prefect to that server (only needed once)
   ```
   prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```
3. Run the ETL flow (first run processes all 24 files)
   ```
   python flows/etl_flow.py
   ```
   The default is configuration A (2 workers with 6 GB each). To run with configuration B instead
   ```
   python flows/etl_flow.py --dask-config B_4workers_3GB
   ```
4. Run the Dask benchmark (uses the cleaned data from step 3)
   ```
   python src/dask_benchmark.py --repeats 2
   ```
5. Open `notebooks/01_eda.ipynb` in VS Code and run all cells.
6. Start Grafana (Docker Desktop must be running)
   ```
   cd grafana
   ```
   ```
   docker compose --env-file ../.env up -d
   ```
   Open http://localhost:3000 (admin / admin). The dashboard is under *Dashboards > Big Data Project*.

### Demonstrating the recurring flow

Run the flow a second time: it reports *no new files*. Then add a new month (for example `python src/download_tlc.py --year 2026 --month 1`) and run the flow again: only the new files are cleaned, checked, analyzed and loaded. Every run is visible in the Prefect UI and in the table `etl_run_log`.

## Things to explain in the report

* **Cleaning.** Negative fares (refunds), zero distance, durations under 1 minute or over 3 hours and speeds above 80 mph are removed. The share per rule is stored in `quality_check` and shown in the notebook.
* **Distribution check.** For every source-month the pipeline also stores the pickup-zone counts and the fare / distance / duration histograms **before and after** cleaning (`*_quality.json`), so the notebook (section 3) can verify that the cleaning did not change the geography of each service (share of trips per pickup borough, taxi share per borough) nor the centre of the numeric distributions — only the invalid tails. Two quality rules enforce this per file: the zone shares may shift by a total variation distance of at most 0.02 and any median by at most 5 % (thresholds in `config.py`). Files cleaned before this check existed must be reprocessed once (`python flows/etl_flow.py --full`).
* **Tips.** Yellow Taxi cash tips are not recorded, so tips are only compared on card payments. Uber and Lyft are always paid in the app.
* **Fares.** Comparisons use the base fare before tips, taxes and surcharges, because the surcharges differ by law between taxis and app services.
* **Congestion pricing.** Since 5 January 2025 trips into the Manhattan congestion relief zone pay a toll (`cbd_congestion_fee`).

## Testing without the real data

`tests/make_synthetic_data.py` generates small fake files in the exact TLC schemas. They were only used to test the code, never for the analysis. Set `DATA_DIR` to a test folder first, so the real data folder is not touched.

## Data source

NYC Taxi & Limousine Commission, *TLC Trip Record Data*, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
