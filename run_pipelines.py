#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipelines.py

Orchestrates updates:
- Detects start_date from the final CSV (last date) and end_date as yesterday (UTC).
- Runs Tides, NCEI, GOES (and optionally Chlorophyll) from [start..yesterday].
- Merges available daily CSVs into a single final CSV.
- Idempotent (dedup by 'date').

Usage:
    python run_pipelines.py --final data/final/SJL_daily_df.csv --default-start 2020-01-01
"""

import os
import sys
import glob
import subprocess
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

# ------------------------------- CONFIG ---------------------------------

# Final combined CSV you maintain
DEFAULT_FINAL_CSV = "data/final/SJL_daily_df.csv"

# Default start if final CSV does not exist or has no valid 'date'
DEFAULT_START_DATE = "2022-01-01"

# Yesterday resolver (UTC)
def yesterday_utc_date() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y-%m-%d")

# TIDES paths & defaults
TIDES_STATION = "9755371"
TIDES_OUT_DIR = "data/tide_data"
TIDES_MERGED_DAILY = "data/combined/tide_data.csv"
TIDES_LOG_DIR = "logs"
TIDES_LOG_FILE = "tides_processed_log.csv"

# NCEI paths & defaults
NCEI_STATION = "GHCND:USW00022521"
NCEI_OUT_DIR = "data/ncei_data"
NCEI_MERGED_DAILY = "data/combined/ncei_daily.csv"
NCEI_LOG_DIR = "logs"
NCEI_LOG_FILE = "ncei_processed_log.csv"
NCEI_DATA_TYPES = ["TMAX", "TMIN", "PRCP", "AWND", "WSF2"]

# GOES / INSOLRICO paths & defaults
GOES_CACHE_DIR = "data/goes_data/cache"
GOES_COMBINED = "data/goes_data/INSOLRICO_SanJoseLake_combined.csv"
GOES_LOG = "logs/processed_lo_goes.csv"
GOES_LAT = 18.425
GOES_LON = -66.025
GOES_RADIUS_KM = 3.0

# CHLOROPHYLL (optional; not merged here by default because it’s scene-level)
CHL_ENABLED = False
CHL_OUT_DIR = "data/chl_data"
CHL_LOG_DIR = "logs"
CHL_LOG_FILE = "chl_processed_log.csv"

# ------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("runner")


# ---------------------------- DATE HELPERS -------------------------------

def parse_ymd(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def infer_start_from_final(final_csv: str, default_start: str) -> str:
    """
    If final_csv exists, read the 'date' column and return the MAX(date) as start.
    Otherwise return default_start.

    Starting from the last existing date is safe because the final merge
    deduplicates by 'date' (keep='last').
    """
    if not os.path.exists(final_csv):
        logger.info(f"Final CSV not found -> using default start: {default_start}")
        return default_start

    try:
        df = pd.read_csv(final_csv)
        if "date" not in df.columns:
            logger.warning(f"'date' column not found in {final_csv}; using default start.")
            return default_start
        d = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna()
        if d.empty:
            logger.info(f"No valid dates in {final_csv}; using default start {default_start}")
            return default_start
        last = max(d)  # start at the last date (no +1)
        logger.info(f"Inferred start from final CSV (using last date): {last}")
        return last.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Could not read {final_csv} ({e}); using default start {default_start}")
        return default_start

def clamp_range(start: str, end: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensure start <= end. If start > end, return (None, None) to skip work.
    """
    ys = parse_ymd(start)
    ye = parse_ymd(end)
    if ys > ye:
        return None, None
    return start, end


# ---------------------------- TIDES RUNNER -------------------------------

def run_tides(start: str, end: str) -> Optional[str]:
    """
    Runs tides pipeline month-by-month, merges to daily, writes TIDES_MERGED_DAILY.
    Returns path to daily CSV or None if failed.
    """
    # Try import first
    try:
        from tides_pipeline import run_multi_month_download, merge_products_daily_avg  # user's file
        os.makedirs(TIDES_LOG_DIR, exist_ok=True)
        log_csv_path = os.path.join(TIDES_LOG_DIR, TIDES_LOG_FILE)

        run_multi_month_download(
            start=start,
            end=end,
            station_id=TIDES_STATION,
            products=["air_temperature", "water_temperature", "air_pressure", "water_level"],
            output_dir=TIDES_OUT_DIR,
            log_csv_path=log_csv_path,
            force=False,
            throttle_s=0.2
        )
        merged = merge_products_daily_avg(
            output_dir=TIDES_OUT_DIR,
            station_id=TIDES_STATION,
            save_path=TIDES_MERGED_DAILY
        )
        return merged or (TIDES_MERGED_DAILY if os.path.exists(TIDES_MERGED_DAILY) else None)
    except Exception as e:
        logger.warning(f"Tides import-run failed, trying CLI. ({e})")

    # Fallback CLI
    try:
        os.makedirs(os.path.dirname(TIDES_MERGED_DAILY), exist_ok=True)
        subprocess.check_call([
            sys.executable, "tides_pipeline.py",
            "--start", start, "--end", end,
            "--station", TIDES_STATION,
            "--out", TIDES_OUT_DIR,
            "--merged", TIDES_MERGED_DAILY,
            "--log-dir", TIDES_LOG_DIR, "--file-log", "--log-level", "INFO"
        ])
        return TIDES_MERGED_DAILY if os.path.exists(TIDES_MERGED_DAILY) else None
    except Exception as e:
        logger.error(f"Tides CLI failed: {e}")
        return None


# ----------------------------- NCEI RUNNER -------------------------------

def run_ncei(start: str, end: str) -> Optional[str]:
    """
    Runs NCEI downloader for [start..end]. Writes/returns a daily CSV.
    Tries modern 'ncei_downloader.run_downloader' first; falls back to older ncei_pipeline API or CLI.
    """
    try:
        from ncei_pipeline import load_ncei_token, download_ncei_data
        os.makedirs(NCEI_OUT_DIR, exist_ok=True)
        token = os.getenv("NCEI_TOKEN") or load_ncei_token(".ncei_token")
        df = download_ncei_data(
            api_token=token,
            start_date=start,
            end_date=end,
            station_id=NCEI_STATION,
            datasetid="GHCND",
            save_csv=True,
            output_dir=NCEI_OUT_DIR,
            max_retries=5
        )
        # Unify into a fixed combined path
        if df is not None and not df.empty:
            os.makedirs(os.path.dirname(NCEI_MERGED_DAILY), exist_ok=True)
            if os.path.exists(NCEI_MERGED_DAILY):
                base = pd.read_csv(NCEI_MERGED_DAILY, parse_dates=["date"])
                out = (pd.concat([base, df], ignore_index=True)
                         .drop_duplicates(subset=["date"], keep="last")
                         .sort_values("date"))
            else:
                out = df.sort_values("date")
            out.to_csv(NCEI_MERGED_DAILY, index=False)
            return NCEI_MERGED_DAILY
        return None
    except Exception as e:
        logger.info(f"ncei_pipeline import path failed ({e}). Trying CLI...")

    # CLI fallback
    try:
        os.makedirs(NCEI_OUT_DIR, exist_ok=True)
        subprocess.check_call([
            sys.executable, "ncei_pipeline.py"
        ])
        cand = sorted(glob.glob(os.path.join(NCEI_OUT_DIR, "*.csv")))
        return cand[-1] if cand else None
    except Exception as e:
        logger.error(f"NCEI CLI failed: {e}")
        return None


# ------------------------------ GOES RUNNER ------------------------------

def run_goes(start: str, end: str) -> Optional[str]:
    """
    Runs the INSOLRICO/GOES downloader for [start..end]. Appends to GOES_COMBINED.
    """
    try:
        os.makedirs(os.path.dirname(GOES_COMBINED), exist_ok=True)
        subprocess.check_call([
            sys.executable, "goes_pipeline.py",
            "--start", start, "--end", end,
            "--cache", GOES_CACHE_DIR,
            "--combined", GOES_COMBINED,
            "--log", GOES_LOG,
            "--lat", str(GOES_LAT),
            "--lon", str(GOES_LON),
            "--radius", str(GOES_RADIUS_KM),
            "--report-missing"
        ])
        return GOES_COMBINED if os.path.exists(GOES_COMBINED) else None
    except Exception as e:
        logger.error(f"GOES run failed: {e}")
        return None


# --------------------------- CHLORO RUNNER (opt) ------------------------

def run_chlorophyll(start: str, end: str) -> Optional[str]:
    """
    Runs chlorophyll pipeline for [start..end]. Produces scene CSVs, not a daily file.
    Returns directory path or None.
    """
    if not CHL_ENABLED:
        return None
    try:
        try:
            from chlorophyll_downloader import main as chl_main
            args = [
                "--start", start, "--end", end,
                "--lon", "-117.31646", "--lat", "32.92993", "--radius-km", "5",
                "--collections", "EO:EUM:DAT:0407", "EO:EUM:DAT:0556",
                "--out", CHL_OUT_DIR,
                "--log-dir", CHL_LOG_DIR, "--log-csv", CHL_LOG_FILE
            ]
            chl_main(args)  # type: ignore
        except Exception:
            subprocess.check_call([
                sys.executable, "chll_pipeline.py",
                "--start", start, "--end", end,
                "--lon", "-117.31646", "--lat", "32.92993", "--radius-km", "5",
                "--collections", "EO:EUM:DAT:0407", "EO:EUM:DAT:0556",
                "--out", CHL_OUT_DIR,
                "--log-dir", CHL_LOG_DIR, "--log-csv", CHL_LOG_FILE
            ])
        return CHL_OUT_DIR
    except Exception as e:
        logger.error(f"Chlorophyll run failed: {e}")
        return None


# ------------------------------ MERGER ----------------------------------

def load_daily_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    """
    Read a daily CSV if present. Returns DataFrame with 'date' normalized to date type.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if date_col not in df.columns:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        return df.dropna(subset=[date_col])
    return pd.DataFrame()

def merge_sources_to_final(final_csv: str,
                           tides_csv: Optional[str],
                           ncei_csv: Optional[str],
                           goes_csv: Optional[str],
                           start: str,
                           end: str) -> str:
    """
    Merge available daily CSVs on 'date' (outer join), then append to final CSV,
    dedup by date, sort ascending, and write back.
    Only rows within [start..end] are merged (safety).
    """
    tides_df = load_daily_csv(tides_csv) if tides_csv else pd.DataFrame()
    ncei_df  = load_daily_csv(ncei_csv)  if ncei_csv  else pd.DataFrame()
    goes_df  = load_daily_csv(goes_csv)  if goes_csv  else pd.DataFrame()

    s = parse_ymd(start)
    e = parse_ymd(end)
    for df in (tides_df, ncei_df, goes_df):
        if not df.empty:
            df.query(" @s <= date <= @e ", inplace=True)

    merged = None
    for df in (tides_df, ncei_df, goes_df):
        if df is None or df.empty:
            continue
        merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")

    if merged is None:
        logger.info("No new source data to merge.")
        return final_csv

    if os.path.exists(final_csv):
        base = pd.read_csv(final_csv)
        base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.date
        merged_all = (pd.concat([base, merged], ignore_index=True)
                        .drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date"))
    else:
        os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)
        merged_all = merged.sort_values("date")

    merged_all.to_csv(final_csv, index=False)
    logger.info(f"Final CSV updated: {final_csv} (rows={len(merged_all)})")
    return final_csv


# ------------------------------- MAIN -----------------------------------

def main():
    p = argparse.ArgumentParser(description="Run pipelines from the last date in final CSV to yesterday (UTC), then update the final CSV.")
    p.add_argument("--final", default=DEFAULT_FINAL_CSV, help="Final combined CSV path")
    p.add_argument("--default-start", default=DEFAULT_START_DATE, help="Start date if final CSV does not exist (YYYY-MM-DD)")
    p.add_argument("--end", default=yesterday_utc_date(), help="End date (YYYY-MM-DD), defaults to yesterday UTC")
    p.add_argument("--skip-tides", action="store_true", help="Skip tides pipeline")
    p.add_argument("--skip-ncei", action="store_true", help="Skip NCEI pipeline")
    p.add_argument("--skip-goes", action="store_true", help="Skip GOES/INSOLRICO pipeline")
    p.add_argument("--skip-chl", action="store_true", help="Skip Chlorophyll pipeline")
    args = p.parse_args()

    # Determine start..end
    start_auto = infer_start_from_final(args.final, args.default_start)
    start, end = clamp_range(start_auto, args.end)
    if start is None:
        logger.info("Nothing to do (final CSV already up to date).")
        return

    logger.info(f"Running pipelines for range: {start} .. {end}")

    tides_csv = ncei_csv = goes_csv = None

    if not args.skip_tides:
        tides_csv = run_tides(start, end)
        logger.info(f"Tides daily: {tides_csv}")

    if not args.skip_ncei:
        ncei_csv = run_ncei(start, end)
        logger.info(f"NCEI daily: {ncei_csv}")

    if not args.skip_goes:
        goes_csv = run_goes(start, end)
        logger.info(f"GOES daily/combined: {goes_csv}")

    if not args.skip_chl and CHL_ENABLED:
        run_chlorophyll(start, end)  # not merged by default

    # Merge
    merge_sources_to_final(args.final, tides_csv, ncei_csv, goes_csv, start, end)


if __name__ == "__main__":
    main()
