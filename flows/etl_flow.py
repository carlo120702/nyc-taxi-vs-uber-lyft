"""Prefect ETL flow: recurring and scalable analytics for NYC Yellow Taxi vs. Uber/Lyft.

What it does on every run
  1. discover   finds monthly raw files (yellow, fhvhv) in data/raw that are not yet in the database
  2. context    loads the taxi zone lookup and daily weather (additional context data)
  3. clean      harmonises both schemas and cleans the trips, in parallel on a Dask cluster
  4. validate   runs data quality rules per source and month (stored in table quality_check)
  5. analyze    builds the KPI tables with Dask (only for the new files)
  6. load       writes the results to AWS RDS PostgreSQL via SQLAlchemy ORM
  7. log        stores the execution in table etl_run_log

If new data arrives (e.g. a new month from TLC), run src/download_tlc.py (or copy the
Parquet file into data/raw) and run the flow again: only the new files are processed.

Usage
    python flows/etl_flow.py                         # incremental run
    python flows/etl_flow.py --full                  # reprocess everything
    python flows/etl_flow.py --dask-config A_2workers_6GB
"""
import argparse
import sys
import time
from pathlib import Path

from dask.distributed import Client, LocalCluster
from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT / "src")])
import config  # noqa: E402
import context  # noqa: E402
import db  # noqa: E402
import pipeline  # noqa: E402


@task(name="discover-new-files")
def discover(full: bool) -> list[tuple[str, int, int]]:
    logger = get_run_logger()
    available = set(pipeline.available_raw())
    keys = sorted(available) if full else sorted(available - db.loaded_keys(db.get_engine()))
    logger.info(f"{len(available)} source-months available, {len(keys)} to process: {keys}")
    return keys


@task(name="load-context-data", retries=1)
def load_context(keys: list[tuple[str, int, int]]):
    years = [y for _, y, _ in keys]
    zones = context.zone_lookup()
    weather = context.weather_daily(min(years), max(years))
    get_run_logger().info(f"zones: {len(zones)} | weather days: {0 if weather is None else len(weather)}")
    return zones, weather


@task(name="clean-and-harmonise-dask", retries=1)
def clean(keys: list[tuple[str, int, int]], overwrite: bool) -> list[str]:
    return [str(p) for p in pipeline.clean_many_with_dask(keys, overwrite=overwrite)]


@task(name="data-quality-checks")
def validate(keys: list[tuple[str, int, int]]) -> list[dict]:
    logger = get_run_logger()
    checks = [c for k in keys for c in pipeline.quality_checks(*k)]
    failed = [c for c in checks if not c["passed"]]
    logger.info(f"{len(checks)} checks executed, {len(failed)} failed")
    for c in failed:
        logger.warning(f"FAILED {c['source']} {c['year']}-{c['month']:02d} {c['check_name']}: {c['value']:.4f}")
    return checks


@task(name="build-aggregates-dask")
def analyze(keys: list[tuple[str, int, int]], zones, weather) -> dict:
    tables = pipeline.build_aggregates(pipeline.read_trips(keys), zones, weather)
    for name, df in tables.items():   # local copy as evidence for the report
        df.to_csv(config.OUTPUT_DIR / f"{name}_last_run.csv", index=False)
    return tables


@task(name="load-to-aws-rds", retries=2, retry_delay_seconds=10)
def load(tables: dict) -> dict:
    return db.load_results(db.get_engine(), tables)


@flow(name="nyc-taxi-vs-uber-lyft-etl", log_prints=True)
def nyc_trips_etl(full: bool = False, dask_config: str = config.DEFAULT_DASK_CONFIG):
    logger = get_run_logger()
    engine = db.get_engine()
    db.create_tables(engine)

    keys = discover(full)
    if not keys:
        logger.info("No new files. Nothing to do.")
        return

    run_id = db.start_run(engine, dask_config, keys)
    cluster = LocalCluster(**config.DASK_CONFIGS[dask_config])
    client = Client(cluster)
    logger.info(f"Dask cluster {dask_config} started, dashboard: {client.dashboard_link}")
    try:
        t0 = time.perf_counter()
        zones, weather = load_context(keys)
        clean(keys, overwrite=full)
        checks = validate(keys)
        db.save_quality(engine, run_id, checks)
        tables = analyze(keys, zones, weather)
        counts = load(tables)
        n_trips = int(tables["service_month_kpi"]["trips"].sum())
        msg = f"rows loaded {counts}; {time.perf_counter() - t0:.1f}s"
        db.finish_run(engine, run_id, "success", n_trips, msg)
        logger.info(f"Finished: {n_trips:,} trips in {len(keys)} source-months. {msg}")
    except Exception as exc:
        db.finish_run(engine, run_id, "failed", message=str(exc)[:2000])
        raise
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="reprocess all files")
    parser.add_argument("--dask-config", default=config.DEFAULT_DASK_CONFIG, choices=list(config.DASK_CONFIGS))
    args = parser.parse_args()
    nyc_trips_etl(full=args.full, dask_config=args.dask_config)
