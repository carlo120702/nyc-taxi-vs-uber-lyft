"""Database layer: SQLAlchemy ORM models for the result tables in AWS RDS
(PostgreSQL) and functions to load the analytics results.

Only aggregated results are stored in AWS, never the raw trip records.
"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (BigInteger, Boolean, Date, DateTime, Float, Integer, SmallInteger, String,
                        Text, create_engine, delete, insert, tuple_)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class Base(DeclarativeBase):
    pass


class KpiColumns:
    """Columns shared by all KPI tables (sums allow correct re-aggregation in Grafana)."""
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(SmallInteger, index=True)
    month: Mapped[int] = mapped_column(SmallInteger, index=True)
    source: Mapped[str] = mapped_column(String(10), index=True)      # yellow | fhvhv
    service: Mapped[str] = mapped_column(String(20), index=True)     # Yellow Taxi | Uber | Lyft
    trips: Mapped[int] = mapped_column(BigInteger)
    revenue: Mapped[float] = mapped_column(Float)                    # total paid by passengers (USD)
    fare_sum: Mapped[float] = mapped_column(Float)
    miles_sum: Mapped[float] = mapped_column(Float)
    minutes_sum: Mapped[float] = mapped_column(Float)
    tip_sum: Mapped[float] = mapped_column(Float)
    avg_fare: Mapped[float] = mapped_column(Float)
    avg_miles: Mapped[float] = mapped_column(Float)
    avg_duration_min: Mapped[float] = mapped_column(Float)
    fare_per_mile: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_speed_mph: Mapped[float | None] = mapped_column(Float, nullable=True)


class ServiceMonthKPI(KpiColumns, Base):
    """Monthly KPIs per service (Yellow Taxi, Uber, Lyft)."""
    __tablename__ = "service_month_kpi"
    tipped_trips: Mapped[int] = mapped_column(BigInteger)
    card_trips: Mapped[int] = mapped_column(BigInteger)
    card_fare_sum: Mapped[float] = mapped_column(Float)
    card_tip_sum: Mapped[float] = mapped_column(Float)
    cbd_trips: Mapped[int] = mapped_column(BigInteger)
    cbd_fee_sum: Mapped[float] = mapped_column(Float)
    airport_trips: Mapped[int] = mapped_column(BigInteger)
    driver_pay_sum: Mapped[float] = mapped_column(Float)
    wait_min_sum: Mapped[float] = mapped_column(Float)
    wait_trips: Mapped[int] = mapped_column(BigInteger)
    tipped_share: Mapped[float] = mapped_column(Float)
    tip_rate_card: Mapped[float | None] = mapped_column(Float, nullable=True)
    cbd_share: Mapped[float] = mapped_column(Float)
    airport_share: Mapped[float] = mapped_column(Float)
    avg_wait_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    driver_pay_share: Mapped[float | None] = mapped_column(Float, nullable=True)


class ZoneMonthKPI(KpiColumns, Base):
    """Monthly KPIs per pickup zone and service (market share by zone and borough)."""
    __tablename__ = "zone_month_kpi"
    pu_location_id: Mapped[int] = mapped_column(SmallInteger, index=True)
    borough: Mapped[str] = mapped_column(String(30), index=True)
    zone: Mapped[str] = mapped_column(String(80))
    service_zone: Mapped[str] = mapped_column(String(30))
    cbd_trips: Mapped[int] = mapped_column(BigInteger)
    airport_trips: Mapped[int] = mapped_column(BigInteger)


class HourWeekdayKPI(KpiColumns, Base):
    """Demand, price and speed by pickup hour and weekday."""
    __tablename__ = "hour_weekday_kpi"
    day_of_week: Mapped[int] = mapped_column(SmallInteger)          # 0 = Monday
    hour: Mapped[int] = mapped_column(SmallInteger)
    wait_min_sum: Mapped[float] = mapped_column(Float)
    wait_trips: Mapped[int] = mapped_column(BigInteger)
    avg_wait_min: Mapped[float | None] = mapped_column(Float, nullable=True)


class DailyKPI(KpiColumns, Base):
    """Daily KPIs per service, enriched with weather (context data)."""
    __tablename__ = "daily_kpi"
    day: Mapped[int] = mapped_column(SmallInteger)
    date: Mapped[date] = mapped_column(Date, index=True)
    day_of_week: Mapped[int] = mapped_column(SmallInteger)
    cbd_trips: Mapped[int] = mapped_column(BigInteger)
    temp_mean_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    precip_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    snow_cm: Mapped[float | None] = mapped_column(Float, nullable=True)


class QualityCheck(Base):
    """Result of each data quality rule per source and month (traceability)."""
    __tablename__ = "quality_check"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    source: Mapped[str] = mapped_column(String(10))
    year: Mapped[int] = mapped_column(SmallInteger)
    month: Mapped[int] = mapped_column(SmallInteger)
    check_name: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)


class EtlRunLog(Base):
    """One row per execution of the Prefect flow."""
    __tablename__ = "etl_run_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20))
    dask_config: Mapped[str] = mapped_column(String(40))
    months_processed: Mapped[str] = mapped_column(Text)
    n_files: Mapped[int] = mapped_column(Integer)
    n_trips: Mapped[int] = mapped_column(BigInteger, default=0)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)


TABLE_MODELS = {
    "service_month_kpi": ServiceMonthKPI,
    "zone_month_kpi": ZoneMonthKPI,
    "hour_weekday_kpi": HourWeekdayKPI,
    "daily_kpi": DailyKPI,
}


def get_engine():
    url = URL.create("postgresql+psycopg2", username=config.DB_USER, password=config.DB_PASSWORD,
                     host=config.DB_HOST, port=config.DB_PORT, database=config.DB_NAME)
    return create_engine(url, pool_pre_ping=True, connect_args={"sslmode": config.DB_SSLMODE})


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")


def load_results(engine, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Idempotent load: rows of the affected (year, month, source) are deleted first and
    then inserted again, so re-running a month never creates duplicates."""
    counts = {}
    with Session(engine) as session, session.begin():
        for name, df in tables.items():
            model = TABLE_MODELS[name]
            columns = [c.key for c in model.__table__.columns if c.key != "id"]
            df = df[columns]
            keys = list({(int(y), int(m), str(s)) for y, m, s in zip(df["year"], df["month"], df["source"])})
            session.execute(delete(model).where(tuple_(model.year, model.month, model.source).in_(keys)))
            records = _records(df)
            for i in range(0, len(records), 5000):          # chunks keep the RDS round trips small
                session.execute(insert(model), records[i:i + 5000])
            counts[name] = len(df)
    return counts


def start_run(engine, dask_config: str, keys: list[tuple[str, int, int]]) -> int:
    with Session(engine) as session, session.begin():
        run = EtlRunLog(started_at=datetime.now(timezone.utc), status="running",
                        dask_config=dask_config, n_files=len(keys),
                        months_processed=",".join(f"{s} {y}-{m:02d}" for s, y, m in keys))
        session.add(run)
        session.flush()
        return run.id


def finish_run(engine, run_id: int, status: str, n_trips: int = 0, message: str | None = None) -> None:
    with Session(engine) as session, session.begin():
        run = session.get(EtlRunLog, run_id)
        run.finished_at = datetime.now(timezone.utc)
        run.status = status
        run.n_trips = n_trips
        run.duration_s = (run.finished_at - run.started_at).total_seconds()
        run.message = message


def save_quality(engine, run_id: int, checks: list[dict]) -> None:
    with Session(engine) as session, session.begin():
        session.add_all([QualityCheck(run_id=run_id, **c) for c in checks])


def loaded_keys(engine) -> set[tuple[str, int, int]]:
    """(source, year, month) already loaded successfully (used for incremental runs)."""
    with Session(engine) as session:
        rows = session.query(ServiceMonthKPI.source, ServiceMonthKPI.year, ServiceMonthKPI.month).distinct().all()
    return {(str(s), int(y), int(m)) for s, y, m in rows}
