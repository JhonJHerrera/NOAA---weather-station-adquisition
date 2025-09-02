import os
import re
import csv
import gzip
import shutil
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://academic.uprm.edu/hdc/solar/INSOLRICO/"

CONVERSION_FACTOR = 1 / 8.64
COLUMN_NAME = "irradiance_Wm2"
FILENAME_RE = re.compile(r'^INSOLRICO\.(\d{4})(\d{3})\.gz$')

# ---------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------
def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def doy_to_date(year: int, doy: int) -> datetime:
    return datetime(year, 1, 1) + timedelta(days=doy - 1)

# ---------------------------------------------------------------------
# Processed log
# ---------------------------------------------------------------------
def load_processed_log(log_path: str) -> pd.DataFrame:
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame(columns=["date", "filename"])

def save_processed_entry(log_path: str, date_str: str, file_name: str) -> None:
    exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["date", "filename"])
        writer.writerow([date_str, file_name])

# ---------------------------------------------------------------------
# HTTP session with retries
# ---------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "etl-goes/1.0"
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = make_session()

# ---------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------
def get_all_file_links(base_url: str) -> list[str]:
    r = SESSION.get(base_url, timeout=60)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = [a.get('href') for a in soup.find_all('a', href=True)]
    files = [h for h in hrefs if isinstance(h, str) and FILENAME_RE.match(h)]
    return sorted(set(files))

def select_files_in_range(file_links: list[str], start_dt: datetime, end_dt: datetime) -> list[tuple[str, str]]:
    selected = []
    for fn in file_links:
        m = FILENAME_RE.match(fn)
        if not m:
            continue
        year = int(m.group(1))
        doy = int(m.group(2))
        dt = doy_to_date(year, doy)
        if start_dt <= dt <= end_dt:
            selected.append((fn, dt.strftime("%Y-%m-%d")))
    selected.sort(key=lambda x: x[1])
    return selected

# ---------------------------------------------------------------------
# Download and extraction
# ---------------------------------------------------------------------
def download_file(url: str, dest_folder: str) -> str:
    os.makedirs(dest_folder, exist_ok=True)
    local_gz_path = os.path.join(dest_folder, os.path.basename(url))
    if os.path.exists(local_gz_path):
        logger.debug(f"Using cached: {local_gz_path}")
        return local_gz_path
    logger.info(f"Downloading: {url}")
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()
    tmp = local_gz_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, local_gz_path)
    return local_gz_path

def extract_gz_to_txt(gz_path: str) -> str:
    txt_path = gz_path.replace(".gz", "")
    with gzip.open(gz_path, 'rt') as f_in, open(txt_path, 'w') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return txt_path

# ---------------------------------------------------------------------
# Parsing and filtering
# ---------------------------------------------------------------------
def extract_coordinates_to_dataframe(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                m = float(parts[0])
                lat = float(parts[1])
                lon = -float(parts[2])  # invert sign of Y -> longitude
                value = m * CONVERSION_FACTOR
                data.append((lon, lat, value))
            except ValueError:
                continue
    return pd.DataFrame(data, columns=["longitude", "latitude", COLUMN_NAME])

def filter_within_radius(df: pd.DataFrame, center_lat: float, center_lon: float, radius_km: float) -> pd.DataFrame:
    if df.empty:
        return df
    lat = np.radians(df["latitude"].to_numpy())
    lon = np.radians(df["longitude"].to_numpy())
    clat = np.radians(center_lat)
    clon = np.radians(center_lon)
    dphi = lat - clat
    dlmb = lon - clon
    a = np.sin(dphi / 2) ** 2 + np.cos(clat) * np.cos(lat) * np.sin(dlmb / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return df.loc[d <= radius_km].copy()

# ---------------------------------------------------------------------
# Write combined in append mode
# ---------------------------------------------------------------------
def append_to_combined_csv(df: pd.DataFrame, date_str: str, combined_path: str) -> None:
    if df.empty:
        return
    df_out = df.copy()
    df_out.insert(0, "date", date_str)  # put "date" column first
    write_header = not os.path.exists(combined_path)
    df_out.to_csv(combined_path, mode="a", index=False, header=write_header, float_format="%.5f")

# ---------------------------------------------------------------------
# Report missing dates (optional)
# ---------------------------------------------------------------------
def report_missing_dates(start_dt: datetime, end_dt: datetime, have_dates: set[str]) -> list[str]:
    missing = []
    day = start_dt
    while day <= end_dt:
        ds = day.strftime("%Y-%m-%d")
        if ds not in have_dates:
            missing.append(ds)
        day += timedelta(days=1)
    return missing

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(start_date: str,
         end_date: str,
         local_download_dir: str,
         combined_csv_path: str,
         log_csv_path: str,
         center_lat: float,
         center_lon: float,
         radius_km: float,
         report_missing: bool) -> None:

    os.makedirs(local_download_dir, exist_ok=True)
    os.makedirs(os.path.dirname(combined_csv_path) or ".", exist_ok=True)

    start_dt = parse_iso_date(start_date)
    end_dt = parse_iso_date(end_date)

    all_files = get_all_file_links(BASE_URL)
    if not all_files:
        logger.warning("No INSOLRICO files found in index. Exiting.")
        return

    wanted = select_files_in_range(all_files, start_dt, end_dt)
    logger.info(f"Index has {len(all_files)} files; {len(wanted)} match {start_date}..{end_date}.")

    if report_missing:
        have_dates_idx = {ds for _, ds in wanted}
        missing = report_missing_dates(start_dt, end_dt, have_dates_idx)
        if missing:
            logger.info(f"Missing {len(missing)} date(s) in range: {', '.join(missing)}")
        else:
            logger.info("No missing dates in requested range.")

    log_df = load_processed_log(log_csv_path)
    processed_dates = set(log_df["date"]) if not log_df.empty else set()

    for i, (file_name, date_str) in enumerate(wanted, 1):
        logger.info(f"[{i}/{len(wanted)}] {file_name} -> {date_str}")
        gz_path = txt_path = None
        try:
            # If we've already logged that date, skip it
            # if date_str in processed_dates:
            #     logger.debug(f"Date already processed (log): {date_str}")
            #     continue

            url = BASE_URL + file_name
            gz_path = download_file(url, local_download_dir)
            txt_path = extract_gz_to_txt(gz_path)

            df = extract_coordinates_to_dataframe(txt_path)
            if df.empty:
                logger.warning(f"{file_name} parsed to empty DataFrame. Skipping.")
                continue

            df_filtered = filter_within_radius(df, center_lat, center_lon, radius_km)
            if df_filtered.empty:
                logger.info(f"{file_name} has no points within {radius_km} km. Skipping.")
                continue

            append_to_combined_csv(df_filtered, date_str, combined_csv_path)
            logger.info(f"Appended {len(df_filtered)} rows to {combined_csv_path} for {date_str}")
            try:
                if gz_path and os.path.exists(gz_path):
                    os.remove(gz_path)
            except Exception as ce:
                logger.warning(f"Could not remove {gz_path}: {ce}")
            
            save_processed_entry(log_csv_path, date_str, file_name)
            processed_dates.add(date_str)

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

        finally:
            try:
                if txt_path and os.path.exists(txt_path):
                    os.remove(txt_path)
            except Exception as ce:
                logger.warning(f"Cleanup failed for {txt_path}: {ce}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download INSOLRICO and save a single combined CSV filtered by radius.")
    parser.add_argument("--start", default="2024-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-10-05", help="End date YYYY-MM-DD")
    parser.add_argument("--cache", default="goes_data", help="Folder for downloaded .gz files")
    parser.add_argument("--combined", default="data/goes_data/INSOLRICO_SanJoseLake_combined.csv",
                        help="Output path for the combined CSV")
    parser.add_argument("--log",   default="log/processed_lo_goes.csv", help="CSV of processed dates")
    parser.add_argument("--lat",   type=float, default=18.425, help="Center latitude")
    parser.add_argument("--lon",   type=float, default=-66.025, help="Center longitude")
    parser.add_argument("--radius",type=float, default=3.0, help="Radius in km")
    parser.add_argument("--report-missing", action="store_true", help="Report missing dates in the range")

    args = parser.parse_args()

    main(
        start_date=args.start,
        end_date=args.end,
        local_download_dir=args.cache,
        combined_csv_path=args.combined,
        log_csv_path=args.log,
        center_lat=args.lat,
        center_lon=args.lon,
        radius_km=args.radius,
        report_missing=args.report_missing
    )
