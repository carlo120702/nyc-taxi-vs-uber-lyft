"""Dask benchmark: the same processing with two cluster configurations.

Config A  2 workers x 4 threads x 6 GB
Config B  4 workers x 2 threads x 3 GB
(both 8 threads / 12 GB in total, see config.py)

For each configuration and repetition the script measures
  - cleaning     raw -> harmonised, cleaned Parquet (optional, --with-cleaning, slow)
  - eda          descriptive statistics over all trips (describe + value counts)
  - aggregates   the KPI tables that are loaded to AWS (groupby heavy)

Results: outputs/dask_benchmark.csv, outputs/dask_benchmark_summary.csv, outputs/dask_benchmark.png

    python src/dask_benchmark.py --repeats 2
    python src/dask_benchmark.py --repeats 1 --with-cleaning
"""
import argparse
import sys
import time
from pathlib import Path

import dask
import pandas as pd
from dask.distributed import Client, LocalCluster

sys.path.extend([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)])
import config  # noqa: E402
import pipeline  # noqa: E402


def run_eda(ddf):
    return dask.compute(
        ddf[["fare", "trip_miles", "duration_min", "tip"]].describe(),
        ddf["service"].value_counts(),
        ddf.groupby("service")["fare"].mean(),
    )


def benchmark(name: str, settings: dict, repeats: int, with_cleaning: bool) -> list[dict]:
    rows = []
    keys = pipeline.available_raw()
    with LocalCluster(**settings) as cluster, Client(cluster) as client:
        print(f"\n=== {name}: {settings} | dashboard {client.dashboard_link}")
        for r in range(1, repeats + 1):
            times = {}
            if with_cleaning:
                t = time.perf_counter()
                pipeline.clean_many_with_dask(keys, overwrite=True)
                times["cleaning"] = time.perf_counter() - t

            ddf = pipeline.read_trips()
            t = time.perf_counter()
            run_eda(ddf)
            times["eda"] = time.perf_counter() - t

            t = time.perf_counter()
            pipeline.build_aggregates(ddf)
            times["aggregates"] = time.perf_counter() - t

            for step, sec in times.items():
                rows.append({"config": name, "n_workers": settings["n_workers"],
                             "threads_per_worker": settings["threads_per_worker"],
                             "memory_per_worker": settings["memory_limit"],
                             "repeat": r, "step": step, "seconds": round(sec, 2),
                             "n_files": len(keys), "n_partitions": ddf.npartitions})
            print(f"repeat {r}: " + " | ".join(f"{k} {v:.1f}s" for k, v in times.items()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--with-cleaning", action="store_true", help="also time the raw -> clean step")
    a = p.parse_args()

    rows = []
    for name, settings in config.DASK_CONFIGS.items():
        rows += benchmark(name, settings, a.repeats, a.with_cleaning)
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUT_DIR / "dask_benchmark.csv", index=False)

    summary = df.groupby(["config", "step"])["seconds"].median().unstack("step")
    summary["total"] = summary.sum(axis=1)
    print("\nMedian seconds per step:\n", summary.round(2))
    summary.round(2).to_csv(config.OUTPUT_DIR / "dask_benchmark_summary.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = summary.drop(columns="total").T.plot(kind="bar", figsize=(8, 4.5), rot=0)
        ax.set_ylabel("Median runtime (seconds)")
        ax.set_title("Dask cluster configurations: runtime per processing step")
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "dask_benchmark.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
