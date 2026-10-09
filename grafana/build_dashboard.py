"""Builds grafana/dashboards/nyc_taxi_vs_apps_management.json (run once, output is committed).

    python grafana/build_dashboard.py
"""
import json
from pathlib import Path

DS = {"type": "grafana-postgresql-datasource", "uid": "nyctaxi-rds"}
T = "make_date(year, month, 1)"
# Grafana macros split their argument at commas, so $__timeFilter(make_date(year, month, 1))
# would break. The explicit BETWEEN with $__timeFrom() / $__timeTo() avoids that.
TF = f"{T} BETWEEN $__timeFrom() AND $__timeTo()"
SVC = "service IN ($service)"
COLORS = {"Yellow Taxi": "#F2B701", "Uber": "#555555", "Lyft": "#E0007A"}


def share(col="trips"):
    """SQL fragment: share of each service in a sum, one column per service."""
    return ", ".join(
        f"sum({col}) FILTER (WHERE service = '{s}')::float / nullif(sum({col}), 0) AS \"{s}\"" for s in COLORS)


def by_service(expr):
    """SQL fragment: one column per service for an expression on filtered rows."""
    return ", ".join(expr.format(f=f"FILTER (WHERE service = '{s}')") + f" AS \"{s}\"" for s in COLORS)


def color_overrides():
    return [{"matcher": {"id": "byName", "options": s},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]} for s, c in COLORS.items()]


def panel(pid, title, ptype, sql, x, y, w, h, fmt="table", unit=None, decimals=None, options=None,
          custom=None, overrides=None, desc="", minv=None, maxv=None):
    defaults = {}
    if unit:
        defaults["unit"] = unit
    if decimals is not None:
        defaults["decimals"] = decimals
    if custom:
        defaults["custom"] = custom
    if minv is not None:
        defaults["min"] = minv
    if maxv is not None:
        defaults["max"] = maxv
    return {"id": pid, "title": title, "type": ptype, "datasource": DS, "description": desc,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"refId": "A", "datasource": DS, "rawQuery": True, "editorMode": "code",
                         "format": fmt, "rawSql": sql}],
            "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
            "options": options or {}}


STAT = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "colorMode": "value", "graphMode": "none", "textMode": "value"}
BARS = {"drawStyle": "bars", "fillOpacity": 80, "stacking": {"mode": "normal", "group": "A"}}
LINES = {"drawStyle": "line", "lineWidth": 2, "pointSize": 6, "showPoints": "always"}

panels = [
    # ---------------------------------------------------------------- headline KPIs
    panel(1, "Trips in period", "stat",
          f"SELECT sum(trips) AS trips FROM service_month_kpi WHERE {TF} AND {SVC}",
          0, 0, 6, 4, unit="short", decimals=1, options=STAT),
    panel(2, "Yellow Taxi market share (trips)", "stat",
          f"SELECT sum(trips) FILTER (WHERE service = 'Yellow Taxi')::float / nullif(sum(trips), 0) AS share "
          f"FROM service_month_kpi WHERE {TF}",
          6, 0, 6, 4, unit="percentunit", decimals=1, options={**STAT, "colorMode": "background"},
          desc="Share of all Yellow Taxi + Uber + Lyft trips that were served by a yellow taxi."),
    panel(3, "Passenger spending (USD)", "stat",
          f"SELECT sum(revenue) AS revenue FROM service_month_kpi WHERE {TF} AND {SVC}",
          12, 0, 6, 4, unit="currencyUSD", decimals=1, options=STAT,
          desc="Everything passengers paid, including tips, taxes, surcharges and the congestion toll."),
    panel(4, "Yellow Taxi share of passenger spending", "stat",
          f"SELECT sum(revenue) FILTER (WHERE service = 'Yellow Taxi')::float / nullif(sum(revenue), 0) AS share "
          f"FROM service_month_kpi WHERE {TF}",
          18, 0, 6, 4, unit="percentunit", decimals=1, options={**STAT, "colorMode": "background"}),

    # ---------------------------------------------------------------- development over time
    panel(5, "Trips per month by service", "timeseries",
          f"SELECT {T} AS time, {by_service('sum(trips) {f}')} FROM service_month_kpi "
          f"WHERE {TF} AND {SVC} GROUP BY 1 ORDER BY 1",
          0, 4, 12, 8, fmt="time_series", unit="short", custom=BARS, overrides=color_overrides()),
    panel(6, "Market share per month (trips)", "timeseries",
          f"SELECT {T} AS time, {share()} FROM service_month_kpi WHERE {TF} GROUP BY 1 ORDER BY 1",
          12, 4, 12, 8, fmt="time_series", unit="percentunit", custom=LINES, overrides=color_overrides(),
          minv=0, maxv=1),

    # ---------------------------------------------------------------- where and when
    panel(7, "Market share by pickup borough", "barchart",
          f"SELECT borough, {share()} FROM zone_month_kpi WHERE {TF} AND borough NOT IN ('Unknown', 'N/A') "
          f"GROUP BY borough ORDER BY 2 DESC",
          0, 12, 12, 9, unit="percentunit",
          options={"orientation": "horizontal", "stacking": "normal", "showValue": "never", "xField": "borough",
                   "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}},
          custom={"fillOpacity": 85}, overrides=color_overrides(), maxv=1,
          desc="Yellow taxis may only pick up street hails; they dominate where hails are frequent (Manhattan, airports)."),
    panel(8, "Trips by pickup hour", "barchart",
          f"SELECT lpad(hour::text, 2, '0') AS hour, {by_service('sum(trips) {f}')} "
          f"FROM hour_weekday_kpi WHERE {TF} AND {SVC} GROUP BY hour ORDER BY hour",
          12, 12, 12, 9, unit="short",
          options={"xField": "hour", "stacking": "normal", "showValue": "never",
                   "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}},
          custom={"fillOpacity": 85}, overrides=color_overrides()),

    # ---------------------------------------------------------------- price and service quality
    panel(9, "Base fare per mile (USD)", "bargauge",
          f"SELECT service, sum(fare_sum) / nullif(sum(miles_sum), 0) AS fare_per_mile "
          f"FROM service_month_kpi WHERE {TF} AND {SVC} GROUP BY service ORDER BY 2 DESC",
          0, 21, 6, 8, unit="currencyUSD", decimals=2,
          options={"orientation": "horizontal", "displayMode": "basic",
                   "reduceOptions": {"values": True, "calcs": [], "fields": "/^fare_per_mile$/"}},
          desc="Fare before tips, taxes and surcharges divided by miles travelled."),
    panel(10, "Tip rate on card payments", "bargauge",
          f"SELECT service, sum(card_tip_sum) / nullif(sum(card_fare_sum), 0) AS tip_rate "
          f"FROM service_month_kpi WHERE {TF} AND {SVC} GROUP BY service ORDER BY 2 DESC",
          6, 21, 6, 8, unit="percentunit", decimals=1,
          options={"orientation": "horizontal", "displayMode": "basic",
                   "reduceOptions": {"values": True, "calcs": [], "fields": "/^tip_rate$/"}},
          desc="Taxi cash tips are not recorded, so only card payments are compared."),
    panel(11, "Average trip: distance and duration", "table",
          f"SELECT service, sum(miles_sum) / sum(trips) AS avg_miles, sum(minutes_sum) / sum(trips) AS avg_minutes, "
          f"sum(miles_sum) / nullif(sum(minutes_sum) / 60, 0) AS avg_speed_mph, "
          f"sum(airport_trips)::float / sum(trips) AS airport_share, sum(cbd_trips)::float / sum(trips) AS cbd_toll_share "
          f"FROM service_month_kpi WHERE {TF} AND {SVC} GROUP BY service ORDER BY sum(trips) DESC",
          12, 21, 12, 8,
          overrides=[{"matcher": {"id": "byRegexp", "options": "avg_.*"}, "properties": [{"id": "decimals", "value": 1}]},
                     {"matcher": {"id": "byRegexp", "options": ".*_share"},
                      "properties": [{"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1}]}]),

    # ---------------------------------------------------------------- zones where taxis compete
    panel(12, "Pickup zones where Yellow Taxi is strongest (min. 50,000 trips)", "table",
          f"SELECT zone, borough, sum(trips) AS trips, "
          f"sum(trips) FILTER (WHERE service = 'Yellow Taxi')::float / sum(trips) AS taxi_share, "
          f"sum(fare_sum) FILTER (WHERE service = 'Yellow Taxi') / nullif(sum(trips) FILTER (WHERE service = 'Yellow Taxi'), 0) AS taxi_avg_fare, "
          f"sum(fare_sum) FILTER (WHERE service <> 'Yellow Taxi') / nullif(sum(trips) FILTER (WHERE service <> 'Yellow Taxi'), 0) AS app_avg_fare "
          f"FROM zone_month_kpi WHERE {TF} GROUP BY zone, borough HAVING sum(trips) >= 50000 "
          f"ORDER BY taxi_share DESC LIMIT 15",
          0, 29, 14, 10,
          overrides=[{"matcher": {"id": "byName", "options": "taxi_share"}, "properties": [
              {"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1},
              {"id": "custom.cellOptions", "value": {"type": "color-background"}},
              {"id": "color", "value": {"mode": "continuous-YlRd"}}]},
              {"matcher": {"id": "byRegexp", "options": ".*_avg_fare"},
               "properties": [{"id": "unit", "value": "currencyUSD"}, {"id": "decimals", "value": 2}]}]),
    panel(13, "Uber/Lyft: waiting time and driver share", "table",
          f"SELECT service, sum(wait_min_sum) / nullif(sum(wait_trips), 0) AS avg_wait_min, "
          f"sum(driver_pay_sum) / nullif(sum(fare_sum), 0) AS driver_pay_share_of_fare "
          f"FROM service_month_kpi WHERE {TF} AND source = 'fhvhv' GROUP BY service",
          14, 29, 10, 5,
          overrides=[{"matcher": {"id": "byName", "options": "avg_wait_min"},
                      "properties": [{"id": "unit", "value": "m"}, {"id": "decimals", "value": 1}]},
                     {"matcher": {"id": "byName", "options": "driver_pay_share_of_fare"},
                      "properties": [{"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1}]}],
          desc="Not available for yellow taxis (no request time and no driver pay in the taxi meter data)."),
    panel(14, "ETL runs and data quality (Prefect)", "table",
          "SELECT r.id AS run, r.started_at, r.status, r.dask_config, r.n_files, r.n_trips, "
          "round(r.duration_s::numeric, 1) AS seconds, "
          "count(q.id) FILTER (WHERE q.passed) AS checks_passed, count(q.id) FILTER (WHERE NOT q.passed) AS checks_failed "
          "FROM etl_run_log r LEFT JOIN quality_check q ON q.run_id = r.id "
          "GROUP BY r.id ORDER BY r.id DESC LIMIT 10",
          14, 34, 10, 5),

    # ---------------------------------------------------------------- context: weather
    panel(15, "Daily trips and precipitation (context data)", "timeseries",
          f"SELECT date::timestamp AS time, {by_service('sum(trips) {f}')}, max(precip_mm) AS \"Precipitation (mm)\" "
          f"FROM daily_kpi WHERE date BETWEEN $__timeFrom() AND $__timeTo() AND {SVC} GROUP BY date ORDER BY date",
          0, 39, 24, 9, fmt="time_series", unit="short", custom={"drawStyle": "line", "lineWidth": 1},
          overrides=color_overrides() + [{"matcher": {"id": "byName", "options": "Precipitation (mm)"}, "properties": [
              {"id": "custom.axisPlacement", "value": "right"}, {"id": "unit", "value": "lengthmm"},
              {"id": "custom.drawStyle", "value": "bars"}, {"id": "custom.fillOpacity", "value": 40},
              {"id": "color", "value": {"mode": "fixed", "fixedColor": "blue"}}]}]),
]

dashboard = {
    "uid": "nyc-taxi-vs-apps", "title": "NYC Yellow Taxi vs. Uber/Lyft - Management Overview",
    "tags": ["big-data", "nyc-tlc"], "timezone": "utc", "schemaVersion": 39, "version": 1,
    "time": {"from": "2025-01-01T00:00:00Z", "to": "2025-12-31T23:59:59Z"},
    "templating": {"list": [{
        "name": "service", "label": "Service", "type": "query", "datasource": DS,
        "query": "SELECT DISTINCT service FROM service_month_kpi ORDER BY 1",
        "definition": "SELECT DISTINCT service FROM service_month_kpi ORDER BY 1",
        "multi": True, "includeAll": True, "current": {"text": "All", "value": "$__all"}, "refresh": 1}]},
    "panels": panels,
}

out = Path(__file__).resolve().parent / "dashboards" / "nyc_taxi_vs_apps_management.json"
out.write_text(json.dumps(dashboard, indent=2))
print("written", out)
