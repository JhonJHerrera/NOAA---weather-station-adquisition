import os
import csv
import time
import json
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------
# Logging (console by default; file handler added later if requested)
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def add_file_logger(log_dir: str,
                    filename: str = "ncei_downloader.log",
                    level: int = logging.INFO) -> str:
    """
    Adds a FileHandler writing runtime logs into log_dir/filename.
    Returns the full log file path.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    return log_path

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
DEFAULT_DATASET = "GHCND"
DEFAULT_STATION = "GHCND:USW00022521"  # San Juan, PR (TJSJ)
DEFAULT_DTYPES = ["TMAX", "TMIN", "PRCP", "AWND", "WSF2"]
DEFAULT_OUTPUT_DIR = "data/ncei_data"

# Log folder + filenames
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_CSV_NAME = "ncei_processed_log.csv"
DEFAULT_RUNLOG_NAME = "ncei_downloader.log"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def env_or_file_token(env_name: str = "NCEI_TOKEN",
                      token_file: str = ".ncei_token") -> str:
    """
    Load token from environment variable first; if not present, fall back to a file
    located next to this script.
    """
    tok = os.getenv(env_name)
    if tok:
        return tok.strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, token_file)
    if not os.path.exists(path):
        raise RuntimeError(
            f"API token not found. Set {env_name} or create a '{token_file}' file next to this script."
        )
    with open(path, "r") as f:
        return f.read().strip()

# ---------------------------------------------------------------------
# HTTP Session with retries/backoff
# ---------------------------------------------------------------------
def make_session(user_agent: str = "etl-goes/1.0") -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent  # no PII, polite identification
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
# Processed log (idempotency) — stored under log_dir/log_csv_name
# ---------------------------------------------------------------------
def load_processed_log(log_csv_path: str) -> pd.DataFrame:
    if os.path.exists(log_csv_path):
        return pd.read_csv(log_csv_path)
    return pd.DataFrame(columns=["chunk_start", "chunk_end", "station", "status", "rows", "filename"])

def save_processed_entry(log_csv_path: str,
                         chunk_start: str,
                         chunk_end: str,
                         station: str,
                         status: str,
                         rows: int,
                         filename: str = "") -> None:
    exists = os.path.exists(log_csv_path)
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["chunk_start", "chunk_end", "station", "status", "rows", "filename"])
        w.writerow([chunk_start, chunk_end, station, status, rows, filename])

# ---------------------------------------------------------------------
# Download one chunk (with pagination via offset)
# ---------------------------------------------------------------------
def fetch_ncei_chunk(api_token: str,
                     start_date: str,
                     end_date: str,
                     station_id: str,
                     datasetid: str,
                     datatypeids: List[str],
                     throttle_s: float = 0.3,
                     per_page_limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Download a date range (chunk) from NCEI using pagination via 'offset' and
    return a pivoted DataFrame (index=date, columns=datatype).
    """
    headers = {"token": api_token}
    offset = 1
    all_rows = []

    while True:
        params = {
            "datasetid": datasetid,
            "datatypeid": datatypeids,   # list expands to repeated query params
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "limit": per_page_limit,
            "units": "metric",
            "includemetadata": "false",
            "offset": offset,
        }
        try:
            resp = SESSION.get(BASE_URL, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp is not None and resp.status_code == 503:
                logger.warning("Received 503 from NCEI; automatic retries/backoff will apply.")
            else:
                logger.error(f"HTTP error for chunk {start_date}..{end_date}: {e}")
                raise
        except Exception as e:
            logger.error(f"Network error: {e}")
            raise

        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from NCEI.")
            raise

        results = data.get("results", [])
        if not results:
            break

        all_rows.extend(results)
        offset += per_page_limit
        time.sleep(throttle_s)  # be nice to the server

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df_p = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean").reset_index()
    df_p = df_p.sort_values("date")
    return df_p

# ---------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------
def write_year_chunk_csv(df: pd.DataFrame,
                         output_dir: str,
                         station_id: str,
                         start_date: str,
                         end_date: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fname = f"ncei_{station_id.replace(':', '_')}_{start_date}_to_{end_date}.csv"
    path = os.path.join(output_dir, fname)
    df.to_csv(path, index=False)
    return path

def append_or_write_combined(df: pd.DataFrame, combined_path: str) -> None:
    write_header = not os.path.exists(combined_path)
    os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
    df.to_csv(combined_path, mode="a", index=False, header=write_header)

def dedupe_combined_by_date_station(combined_path: str) -> None:
    if not os.path.exists(combined_path):
        return
    df = pd.read_csv(combined_path, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date", "station"], keep="last").sort_values("date")
    df.to_csv(combined_path, index=False)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_downloader(start_date: str,
                   end_date: str,
                   station_id: str,
                   datasetid: str,
                   datatypeids: List[str],
                   output_dir: str,
                   combined_csv_path: Optional[str],
                   log_csv_path: str,
                   token_env: str,
                   token_file: str,
                   per_year: bool,
                   force: bool,
                   overwrite: bool,
                   throttle_s: float,
                   dedupe_after: bool) -> None:

    api_token = env_or_file_token(token_env, token_file)
    os.makedirs(output_dir, exist_ok=True)

    # Default combined path
    if combined_csv_path is None:
        combined_csv_path = os.path.join(
            output_dir, f"ncei_{station_id.replace(':','_')}_{start_date}_to_{end_date}_combined.csv"
        )

    # Overwrite combined if requested
    if overwrite and os.path.exists(combined_csv_path):
        os.remove(combined_csv_path)
        logger.info(f"Removed existing combined CSV: {combined_csv_path}")

    # Year-range
    y0 = parse_iso_date(start_date).year
    y1 = parse_iso_date(end_date).year

    # Log
    log_df = load_processed_log(log_csv_path)

    # Prepare chunks (per-year or single chunk)
    chunks: List[Tuple[str, str]] = []
    if per_year:
        for y in range(y0, y1 + 1):
            s = f"{y}-01-01" if y > y0 else start_date
            e = f"{y}-12-31" if y < y1 else end_date
            chunks.append((s, e))
    else:
        chunks.append((start_date, end_date))

    # Already-processed chunks (idempotent)
    processed = set()
    if not log_df.empty:
        for _, r in log_df.iterrows():
            if r.get("status") in ("ok", "no_data"):
                processed.add((str(r["chunk_start"]), str(r["chunk_end"]), str(r["station"])))

    for i, (s, e) in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] Downloading {s}..{e} for {station_id}")
        if not force and (s, e, station_id) in processed:
            logger.debug(f"Chunk already in log: {s}..{e} ({station_id}). Skipping.")
            continue

        try:
            df = fetch_ncei_chunk(
                api_token=api_token,
                start_date=s,
                end_date=e,
                station_id=station_id,
                datasetid=datasetid,
                datatypeids=datatypeids,
                throttle_s=throttle_s,
            )
        except Exception as ex:
            logger.error(f"Failed to download {s}..{e}: {ex}")
            # Do not log as processed so we can retry later
            continue

        if df is None or df.empty:
            logger.info(f"No data for {s}..{e}.")
            save_processed_entry(log_csv_path, s, e, station_id, status="no_data", rows=0, filename="")
            continue

        # Per-chunk CSV (useful for audit/debug)
        chunk_path = write_year_chunk_csv(df, output_dir, station_id, s, e)

        # Append to combined CSV
        df_copy = df.copy()
        df_copy.insert(1, "station", station_id)
        append_or_write_combined(df_copy, combined_csv_path)

        save_processed_entry(
            log_csv_path, s, e, station_id, status="ok", rows=len(df_copy), filename=os.path.basename(chunk_path)
        )

        time.sleep(throttle_s)  # soft throttle between chunks

    if dedupe_after:
        dedupe_combined_by_date_station(combined_csv_path)
        logger.info(f"Combined CSV deduplicated by ['date', 'station']: {combined_csv_path}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download daily NCEI (NOAA) data with idempotency and chunk logging.")
    p.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default="2025-02-01", help="End date YYYY-MM-DD")
    p.add_argument("--station", default=DEFAULT_STATION, help="NCEI station ID (e.g., GHCND:USW00022521)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset ID (e.g., GHCND)")
    p.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES, help="Datatype IDs (e.g., TMAX TMIN PRCP AWND WSF2)")
    p.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Output folder for per-chunk CSVs & combined CSV")
    p.add_argument("--combined", default=None, help="Path for the combined CSV (optional)")

    # Log folder + files
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Folder to store logs (processed CSV and runtime .log)")
    p.add_argument("--log-csv", default=DEFAULT_LOG_CSV_NAME, help="Processed-chunks log CSV filename (inside log-dir)")
    p.add_argument("--file-log", action="store_true", help="Also write a runtime .log file in log-dir")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Console log level")

    # Behavior toggles
    p.add_argument("--per-year", dest="per_year", action="store_true", default=True,
                   help="Process in yearly chunks (default ON)")
    p.add_argument("--one-chunk", dest="per_year", action="store_false",
                   help="Process the whole range as a single chunk")
    p.add_argument("--force", action="store_true", help="Ignore processed log and reprocess chunks")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the combined CSV if it exists")
    p.add_argument("--throttle", type=float, default=0.3, help="Pause between requests/chunks (seconds)")
    p.add_argument("--dedupe", dest="dedupe_after", action="store_true", default=True,
                   help="Deduplicate combined CSV at the end (default ON)")
    p.add_argument("--no-dedupe", dest="dedupe_after", action="store_false",
                   help="Disable deduplication at the end")

    args = p.parse_args()

    # Configure console log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Compute processed-log CSV path (under log-dir) and ensure directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    processed_log_csv_path = os.path.join(args.log_dir, args.log_csv)

    # Optional runtime file log
    if args.file_log:
        file_log_path = add_file_logger(args.log_dir, DEFAULT_RUNLOG_NAME, level=getattr(logging, args.log_level))
        logger.info(f"Runtime log file: {file_log_path}")

    logger.info(f"Processed-log CSV: {processed_log_csv_path}")

    run_downloader(
        start_date=args.start,
        end_date=args.end,
        station_id=args.station,
        datasetid=args.dataset,
        datatypeids=args.dtypes,
        output_dir=args.out,
        combined_csv_path=args.combined,
        log_csv_path=processed_log_csv_path,
        token_env="NCEI_TOKEN",
        token_file=".ncei_token",
        per_year=args.per_year,
        force=args.force,
        overwrite=args.overwrite,
        throttle_s=args.throttle,
        dedupe_after=args.dedupe_after,
    )
