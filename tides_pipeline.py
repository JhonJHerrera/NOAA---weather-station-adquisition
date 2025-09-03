#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import argparse
import logging
import calendar
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.rrule import rrule, MONTHLY

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def add_file_logger(log_dir: str, filename: str = "tides_pipeline.log", level: int = logging.INFO) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    return path

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DEFAULT_STATION = "9755371"  # San Juan Harbor (example)
DEFAULT_PRODUCTS = ["air_temperature", "water_temperature", "air_pressure", "water_level"]
DEFAULT_INTERVAL = "h"  # apply only when valid
DEFAULT_UNITS = "metric"
DEFAULT_TZ = "gmt"

DEFAULT_OUT_DIR = "data/tide_data"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_CSV = "tides_processed_log.csv"

# Products that typically accept 'datum' and 'interval'
PRODUCTS_NEED_DATUM = {"water_level"}
PRODUCTS_ACCEPT_INTERVAL = {"water_level"}  # expand if needed (e.g., 'wind' uses different semantics)

# ---------------------------------------------------------------------
# HTTP Session with retries/backoff
# ---------------------------------------------------------------------
def make_session(user_agent: str = "etl-goes/1.0") -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET"},
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ---------------------------------------------------------------------
# Processed log (month-level idempotency)
# ---------------------------------------------------------------------
def load_processed_log(log_csv_path: str) -> pd.DataFrame:
    if os.path.exists(log_csv_path):
        return pd.read_csv(log_csv_path)
    return pd.DataFrame(columns=["year", "month", "product", "station", "status", "rows", "filename"])

def save_processed_entry(log_csv_path: str,
                         year: int,
                         month: int,
                         product: str,
                         station: str,
                         status: str,
                         rows: int,
                         filename: str = "") -> None:
    exists = os.path.exists(log_csv_path)
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["year", "month", "product", "station", "status", "rows", "filename"])
        w.writerow([year, month, product, station, status, rows, filename])

# ---------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------
def clamp_month_span(y: int, m: int, start: date, end: date) -> Tuple[str, str]:
    """
    For a given (year, month), return begin/end date strings clamped to the global [start, end] range.
    """
    first = date(y, m, 1)
    last = date(y, m, calendar.monthrange(y, m)[1])
    begin = max(first, start)
    finish = min(last, end)
    return begin.strftime("%Y-%m-%d"), finish.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------
# Download per product and month (clamped to global range)
# ---------------------------------------------------------------------
def download_tide_data(station_id: str,
                       begin_date: str,
                       end_date: str,
                       product: str,
                       interval: Optional[str] = DEFAULT_INTERVAL,
                       units: str = DEFAULT_UNITS,
                       time_zone: str = DEFAULT_TZ,
                       max_retries_local: int = 3,
                       throttle_s: float = 0.0) -> Optional[pd.DataFrame]:
    """
    Fetch one product for [begin_date, end_date], return a DataFrame with columns ['date', product].
    Applies 'datum' and 'interval' only when applicable to the product.
    """
    params = {
        "station": station_id,
        "begin_date": begin_date,
        "end_date": end_date,
        "product": product,
        "units": units,
        "time_zone": time_zone,
        "format": "json"
    }
    if product in PRODUCTS_NEED_DATUM:
        params["datum"] = "MLLW"
    if interval and (product in PRODUCTS_ACCEPT_INTERVAL):
        params["interval"] = interval

    logger.info(f"Fetching {product} {begin_date}..{end_date} for station {station_id}")

    for attempt in range(max_retries_local):
        try:
            resp = SESSION.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning(f"{product} {begin_date}..{end_date} error: {data['error'].get('message')}")
                return None
            key = "data" if "data" in data else "predictions" if "predictions" in data else None
            if key is None or not data.get(key):
                return None
            df = pd.DataFrame(data[key])
            # Normalize
            if "t" in df.columns:
                df["date"] = pd.to_datetime(df["t"])
                df.drop(columns=["t"], inplace=True)
            elif "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"])
                df.drop(columns=["time"], inplace=True)

            val_col = "v" if "v" in df.columns else "value" if "value" in df.columns else None
            if val_col is None:
                logger.warning(f"{product} {begin_date}..{end_date}: no value column")
                return None

            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            df = df.rename(columns={val_col: product})
            df = df[["date", product]].dropna(subset=[product])
            time.sleep(throttle_s)
            return df

        except requests.exceptions.HTTPError as e:
            status = getattr(resp, "status_code", None)
            if status == 400:
                logger.error(f"400 for {product} {begin_date}..{end_date}: {e}")
                return None
            if attempt < max_retries_local - 1:
                wait = 2 ** attempt
                logger.warning(f"HTTPError for {product} (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Failed {product} after {max_retries_local} attempts.")
                return None
        except Exception as e:
            if attempt < max_retries_local - 1:
                wait = 2 ** attempt
                logger.warning(f"Error {product} (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Failed {product}: {e}")
                return None

# ---------------------------------------------------------------------
# Main month-loop downloader
# ---------------------------------------------------------------------
def run_multi_month_download(start: str,
                             end: str,
                             station_id: str,
                             products: List[str],
                             output_dir: str,
                             log_csv_path: str,
                             force: bool = False,
                             throttle_s: float = 0.2) -> Dict[str, List[str]]:
    """
    Downloads each product month-by-month (clamped), appends to per-product CSVs,
    logs processed (year, month, product) in log_csv_path, and returns a map of product -> list of CSV paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_df = load_processed_log(log_csv_path)

    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()

    # Build processed set
    processed = set()
    if not log_df.empty:
        for _, r in log_df.iterrows():
            if r.get("status") in ("ok", "no_data"):
                processed.add((int(r["year"]), int(r["month"]), str(r["product"]), str(r["station"])))

    emitted: Dict[str, List[str]] = {p: [] for p in products}

    for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt):
        y, m = dt.year, dt.month
        b, e = clamp_month_span(y, m, start_dt, end_dt)

        for product in products:
            if not force and (y, m, product, station_id) in processed:
                logger.debug(f"Already logged: {y}-{m:02d} {product}. Skipping.")
                continue

            df = download_tide_data(
                station_id=station_id,
                begin_date=b,
                end_date=e,
                product=product,
                interval=DEFAULT_INTERVAL,
                units=DEFAULT_UNITS,
                time_zone=DEFAULT_TZ,
                max_retries_local=3,
                throttle_s=throttle_s
            )

            if df is None or df.empty:
                save_processed_entry(log_csv_path, y, m, product, station_id, status="no_data", rows=0, filename="")
                continue

            # Append to per-product "full" CSV (station-scoped)
            fname = f"{product}_{station_id}_full.csv"
            path = os.path.join(output_dir, fname)
            write_header = not os.path.exists(path)
            df.sort_values("date").to_csv(path, mode="a", index=False, header=write_header)
            emitted[product].append(path)

            save_processed_entry(log_csv_path, y, m, product, station_id, status="ok", rows=len(df), filename=fname)

    return emitted

# ---------------------------------------------------------------------
# Merge & daily averages
# ---------------------------------------------------------------------
def merge_products_daily_avg(output_dir: str,
                             station_id: str,
                             save_path: str) -> Optional[str]:
    """
    Reads all '<product>_<station>_full.csv' in output_dir, computes daily averages per product,
    outer-joins on 'date', saves to save_path, and returns the path.
    """
    from glob import glob

    pattern = os.path.join(output_dir, f"*_{station_id}_full.csv")
    files = glob(pattern)
    if not files:
        logger.warning(f"No per-product CSVs found at {output_dir} for station {station_id}.")
        return None

    merged_df: Optional[pd.DataFrame] = None

    for file in files:
        df = pd.read_csv(file)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        value_col = [c for c in df.columns if c != "date"][0]
        # daily average in GMT (consistent with API time_zone)
        daily = df.groupby(df["date"].dt.date)[value_col].mean().reset_index()
        daily.rename(columns={"date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"])

        if merged_df is None:
            merged_df = daily
        else:
            merged_df = pd.merge(merged_df, daily, on="date", how="outer")

    if merged_df is None or merged_df.empty:
        logger.warning("Nothing to merge.")
        return None

    merged_df = merged_df.sort_values("date")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    logger.info(f"Saved daily averaged merged data to {save_path}")
    return save_path

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="NOAA Tides & Currents monthly downloader with idempotent logging and daily-avg merge.")
    p.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default="2025-07-20", help="End date YYYY-MM-DD")
    p.add_argument("--station", default=DEFAULT_STATION, help="NOAA CO-OPS station ID (e.g., 9755371)")
    p.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS, help="Products to fetch")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory for per-product CSVs")
    p.add_argument("--merged", default="data/tide_data/tide_data.csv", help="Path for merged daily-avg CSV")

    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for logs")
    p.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help="Processed log CSV filename (inside log-dir)")
    p.add_argument("--file-log", action="store_true", help="Write runtime .log file under log-dir")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Console log level")

    p.add_argument("--force", action="store_true", help="Reprocess months even if logged")
    p.add_argument("--throttle", type=float, default=0.2, help="Pause between product requests (seconds)")

    args = p.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    os.makedirs(args.log_dir, exist_ok=True)
    log_csv_path = os.path.join(args.log_dir, args.log_csv)

    if args.file_log:
        path = add_file_logger(args.log_dir, level=getattr(logging, args.log_level))
        logger.info(f"Runtime log file at: {path}")

    emitted = run_multi_month_download(
        start=args.start,
        end=args.end,
        station_id=args.station,
        products=args.products,
        output_dir=args.out,
        log_csv_path=log_csv_path,
        force=args.force,
        throttle_s=args.throttle
    )

    merge_products_daily_avg(
        output_dir=args.out,
        station_id=args.station,
        save_path=args.merged
    )

if __name__ == "__main__":
    main()
