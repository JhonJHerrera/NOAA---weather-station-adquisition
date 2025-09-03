#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from shapely import geometry

import eumdac

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def add_file_logger(log_dir: str, filename: str = "chlorophyll_pipeline.log", level: int = logging.INFO) -> str:
    """
    Add a FileHandler that writes runtime logs into log_dir/filename.
    Returns the full path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    return path

# ---------------------------------------------------------------------
# Defaults / Constants
# ---------------------------------------------------------------------
DEFAULT_OUT_DIR = "data/chl_data"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_CSV = "chl_processed_log.csv"

DEFAULT_PRODUCT_PATTERNS = [
    "chl_nn", "oa08_reflectance", "oa11_reflectance", "oa17_reflectance",
    "iwv", "oa04_reflectance", "oa21_reflectance", "oa07_reflectance",
    "chl_oc4me", "t865", "a865", "tsm_nn", "oa12_reflectance", "oa03_reflectance",
    "oa16_reflectance", "oa01_reflectance", "oa06_reflectance", "par",
    "adg443_nn", "kd490_m07", "oa02_reflectance", "oa09_reflectance",
    "oa05_reflectance", "oa10_reflectance", "oa18_reflectance",
    # required by the pipeline
    "geo_coordinates", "wqsf",
]

# Never download entries that contain these tokens (lowercase, substring match)
EXCLUDED_PATTERNS = {"tie_geo_coordinates"}

MAX_RETRIES_DOWNLOAD = 3
RETRY_DELAY_START = 5  # seconds
RETRY_DELAY_MAX = 60   # seconds

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def km_to_degrees(km: float) -> float:
    """
    Rough conversion: ~111.32 km per degree latitude.
    Good enough for a square ROI around a small area.
    """
    return km / 111.32

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path or ".", exist_ok=True)

def normalize_tokens(csv_like: str) -> List[str]:
    """
    Split a comma-separated string into lowercase tokens, trimmed.
    Empty/None returns [].
    """
    if not csv_like:
        return []
    return [t.strip().lower() for t in csv_like.split(",") if t.strip()]

# ---------------------------------------------------------------------
# Processed log (idempotency)
# ---------------------------------------------------------------------
def load_processed_log(log_csv_path: str) -> pd.DataFrame:
    """
    Load the processed-log CSV (if present). Otherwise return an empty DataFrame
    with the expected schema. Ensures the parent directory exists.
    """
    if os.path.exists(log_csv_path):
        return pd.read_csv(log_csv_path)
    ensure_dir(os.path.dirname(log_csv_path))
    return pd.DataFrame(columns=["product_id", "entry_name", "collection_id", "status", "rows", "csv_file"])

def save_processed_entry(log_csv_path: str,
                         product_id: str,
                         entry_name: str,
                         collection_id: str,
                         status: str,
                         rows: int,
                         csv_file: str = "") -> None:
    """
    Append a new row to the processed-log CSV.
    """
    ensure_dir(os.path.dirname(log_csv_path))
    exists = os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["product_id", "entry_name", "collection_id", "status", "rows", "csv_file"])
        w.writerow([product_id, entry_name, collection_id, status, rows, csv_file])

# ---------------------------------------------------------------------
# Timeliness helpers — (prefer NT > STC > NR)
# ---------------------------------------------------------------------
def extract_entry_name_from_pid(pid: str) -> Optional[str]:
    """
    Extract the timestamp-like entry_name from the product_id.
    For S3 OLCI IDs it is typically at index 7 after splitting by '_'.
    """
    parts = str(pid).split("_")
    return parts[7] if len(parts) > 7 else None

def timeliness_rank(pid: str) -> int:
    """
    Rank timeliness from best to worst:
    NT (non-time-critical) > STC/ST > NR/NRT.
    Returns higher number for better timeliness.
    """
    p = pid.upper()
    if "_O_NT_" in p:                       # final
        return 3
    if "_O_STC_" in p or "_O_ST_" in p:     # final-ish
        return 2
    if "_O_NR_" in p or "_O_NRT_" in p:     # near-real-time
        return 1
    return 0

def pick_best_timeliness_per_entry(products) -> list:
    """
    Collapse multiple products with the same entry_name, keeping only the one
    with the best timeliness (highest timeliness_rank).
    """
    best = {}  # entry_name -> (rank, product)
    for prod in products:
        pid = getattr(prod, "_id", str(prod))
        entry = extract_entry_name_from_pid(pid)
        if not entry:
            continue
        r = timeliness_rank(pid)
        prev = best.get(entry)
        if (prev is None) or (r > prev[0]):
            best[entry] = (r, prod)

    # Keep chronological order by entry_name if it looks like YYYYMMDDTHHMMSS
    return [bp[1] for entry, bp in sorted(best.items(), key=lambda kv: kv[0])]

# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------
def build_roi_polygon(lon_center: float, lat_center: float, half_size_deg: float) -> Tuple[List[Tuple[float, float]], str]:
    """
    Build a closed square ROI polygon around (lon_center, lat_center) with +/- half_size_deg.
    Returns the coordinate list and the WKT POLYGON representation.
    """
    roi_coords = [
        (lon_center + half_size_deg, lat_center + half_size_deg),
        (lon_center - half_size_deg, lat_center + half_size_deg),
        (lon_center - half_size_deg, lat_center - half_size_deg),
        (lon_center + half_size_deg, lat_center - half_size_deg),
        (lon_center + half_size_deg, lat_center + half_size_deg),
    ]
    roi_wkt = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in roi_coords]) + "))"
    return roi_coords, roi_wkt

def shutil_copyfileobj(src, dst, length: int = 1024 * 1024) -> None:
    """
    Chunked binary copy (default 1 MB) from a file-like src to a file-like dst.
    """
    while True:
        buf = src.read(length)
        if not buf:
            break
        dst.write(buf)

def safe_download_entry(product, entry: str, dest_dir: str) -> Optional[str]:
    """
    Download a product 'entry' into dest_dir with exponential backoff and a .part temp file.
    Returns the final path, or None on persistent failure.
    """
    filename = os.path.basename(entry)
    final_path = os.path.join(dest_dir, filename)
    tmp_path = final_path + ".part"

    if os.path.exists(final_path):
        return final_path

    delay = RETRY_DELAY_START
    for attempt in range(1, MAX_RETRIES_DOWNLOAD + 1):
        try:
            with product.open(entry=entry) as fsrc, open(tmp_path, "wb") as fdst:
                logger.info(f"Downloading {filename} (attempt {attempt})")
                shutil_copyfileobj(fsrc, fdst)
            os.replace(tmp_path, final_path)
            return final_path
        except Exception as e:
            logger.warning(f"Error downloading {filename}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt < MAX_RETRIES_DOWNLOAD:
                time.sleep(min(delay, RETRY_DELAY_MAX))
                delay *= 2
            else:
                logger.error(f"Persistent error downloading {filename}. Giving up.")
                return None

def find_any(file_map: dict, *candidates: str) -> Optional[str]:
    """
    Case-insensitive lookup: return the first matching file path from file_map
    whose basename equals any candidate (string equality).
    """
    if not file_map:
        return None
    # exact first
    for c in candidates:
        if c in file_map:
            return file_map[c]
    # case-insensitive
    lower_map = {k.lower(): v for k, v in file_map.items()}
    for c in candidates:
        v = lower_map.get(c.lower())
        if v:
            return v
    return None

def process_chlorophyll_data(datastore,
                             lon: float,
                             lat: float,
                             radius_km: float,
                             start_date: str,
                             end_date: str,
                             collection_ids: List[str],
                             output_dir: str,
                             selected_products: List[str],
                             log_csv_path: str,
                             force: bool = False,
                             throttle_s: float = 0.2) -> None:
    """
    Download and process chlorophyll-related satellite products inside an ROI and date range.

    - Idempotent by product_id: skip items already listed in the processed log (unless --force).
    - For each product (scene), write one CSV with columns: latitude, longitude, datetime,
      optional WQSF flags, and additional variables that match the geo grid shape.
    - Option A: for each 'entry_name', keep only the best timeliness (NT > STC > NR).
    """
    ensure_dir(output_dir)

    # Time logging CSV (one row per entry_name)
    time_log_file = os.path.join(output_dir, "time_spent.csv")
    ensure_dir(os.path.dirname(time_log_file))
    with open(time_log_file, "w", newline="") as tf:
        w = csv.writer(tf)
        w.writerow(["entry_name", "seconds"])

    # Load processed-ids (idempotency)
    log_df = load_processed_log(log_csv_path)
    processed_ids = set(log_df["product_id"]) if not log_df.empty else set()

    # Build the wanted pattern list (lowercase) and apply exclusions
    extra_patterns = [p.lower() for p in (selected_products or [])]
    selected_patterns = list(dict.fromkeys(DEFAULT_PRODUCT_PATTERNS + extra_patterns))
    selected_patterns = [p for p in selected_patterns if p not in EXCLUDED_PATTERNS]

    # ROI polygon (square, degrees)
    half_size_deg = km_to_degrees(radius_km)
    roi_coords, roi_wkt = build_roi_polygon(lon, lat, half_size_deg)
    polygon = geometry.Polygon(roi_coords)

    total_time = 0.0

    for collection_id in collection_ids:
        coll = datastore.get_collection(collection_id)
        logger.info(f"Searching products in {collection_id} for {start_date}..{end_date}")
        try:
            products = coll.search(geo=roi_wkt, dtstart=start_date, dtend=end_date)
        except Exception as e:
            logger.error(f"Error searching products in {collection_id}: {e}")
            continue

        # ---------------- Option A: keep only best timeliness per entry ----------------
        products = pick_best_timeliness_per_entry(products)
        # -----------------------------------------------------------------------------

        for product in products:
            product_id = getattr(product, "_id", None) or str(product)
            if (not force) and (product_id in processed_ids):
                logger.debug(f"Already processed: {product_id}. Skipping.")
                continue

            # Derive a timestamp-like entry name from the product_id as best effort
            entry_name = extract_entry_name_from_pid(product_id) or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

            logger.info(f"Processing product {product_id} (entry_name={entry_name})")
            start_t = time.time()
            downloaded_files: List[str] = []

            # 1) Download required entries matching our patterns (case-insensitive, substring)
            try:
                for entry in product.entries:
                    entry_base = os.path.basename(entry)
                    entry_base_l = entry_base.lower()

                    # exclude unwanted entries
                    if any(ex in entry_base_l for ex in EXCLUDED_PATTERNS):
                        logger.info(f"Skipping excluded entry: {entry_base}")
                        continue

                    # include if any wanted token is present
                    if not any(pat in entry_base_l for pat in selected_patterns):
                        continue

                    path = safe_download_entry(product, entry, output_dir)
                    if path:
                        downloaded_files.append(path)

                    if throttle_s > 0:
                        time.sleep(throttle_s)
            except Exception as e:
                logger.error(f"Error while downloading entries for {product_id}: {e}")
                _cleanup_files(downloaded_files)
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "download_error", 0, "")
                continue

            # 2) Map essential files; be tolerant to case
            file_map = {os.path.basename(f): f for f in downloaded_files}
            geo_path  = find_any(file_map, "geo_coordinates.nc")
            flag_path = find_any(file_map, "wqsf.nc")
            # main chlorophyll variables (we'll accept either; you can extend list if needed)
            chl_nn_path    = find_any(file_map, "chl_nn.nc")
            chl_oc4me_path = find_any(file_map, "chl_oc4me.nc")
            chl_paths = [p for p in (chl_nn_path, chl_oc4me_path) if p]

            if (geo_path is None) or (not chl_paths):
                logger.warning("Missing required files (geo_coordinates or chlorophyll). Skipping scene.")
                _cleanup_files(downloaded_files)
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "missing_files", 0, "")
                continue

            # 3) Process and write CSV
            csv_out = os.path.join(output_dir, f"{entry_name}.csv")
            rows_written = 0
            try:
                # Load geolocation arrays
                with xr.open_dataset(geo_path) as geo_ds:
                    lat_arr = geo_ds["latitude"].data
                    lon_arr = geo_ds["longitude"].data

                # Build ROI mask over the geolocation grid
                flat_lon = lon_arr.flatten()
                flat_lat = lat_arr.flatten()
                mask_flat = np.array([
                    polygon.contains(geometry.Point(x, y))
                    for x, y in zip(flat_lon, flat_lat)
                ])
                mask = mask_flat.reshape(lat_arr.shape)

                # Safety: if shapes don't match, write an empty mask
                if lat_arr.shape != mask.shape:
                    logger.warning(f"Mask shape mismatch for {entry_name}; writing empty mask.")
                    mask = np.zeros_like(lat_arr, dtype=bool)

                # Scene timestamp (same for all rows)
                try:
                    ts = datetime.strptime(entry_name, "%Y%m%dT%H%M%S")
                except Exception:
                    ts = datetime.utcnow()

                df = pd.DataFrame({
                    "latitude":  lat_arr[mask],
                    "longitude": lon_arr[mask],
                    "datetime":  ts
                })

                # Optional: WQSF flags
                if flag_path and os.path.exists(flag_path):
                    with xr.open_dataset(flag_path) as flag_ds:
                        wqsf = flag_ds["WQSF"].data
                    if wqsf.shape == lat_arr.shape:
                        wqsf_masked = wqsf[mask]
                        df["INVALID"] = (wqsf_masked & (1 << 0)) > 0
                        df["WATER"]   = (wqsf_masked & (1 << 1)) > 0
                        df["CLOUD"]   = (wqsf_masked & (1 << 2)) > 0
                        df["LAND"]    = (wqsf_masked & (1 << 3)) > 0
                    else:
                        logger.warning(f"WQSF shape mismatch for {entry_name}. Skipping flags.")

                # Track all variable names we've written (helps auditing)
                var_list_file = os.path.join(output_dir, "var_names.txt")
                seen_vars = set()
                if os.path.exists(var_list_file):
                    with open(var_list_file, "r") as f:
                        seen_vars = {line.strip() for line in f if line.strip()}

                # Add any additional .nc variables whose arrays match the geo grid shape
                for fpath in downloaded_files:
                    base = os.path.basename(fpath).lower()
                    if fpath.endswith(".nc") and base not in {"geo_coordinates.nc", "wqsf.nc"}:
                        try:
                            with xr.open_dataset(fpath) as ds:
                                for var_name, da in ds.data_vars.items():
                                    v = da.data
                                    if hasattr(v, "shape") and v.shape == lat_arr.shape:
                                        df[var_name] = v[mask]
                                        seen_vars.add(var_name)
                        except Exception as e:
                            logger.warning(f"Error reading {fpath}: {e}")

                # Persist the variable name list (sorted for readability)
                with open(var_list_file, "w") as f:
                    for v in sorted(seen_vars):
                        f.write(v + "\n")

                # Save scene CSV
                df.to_csv(csv_out, index=False)
                rows_written = len(df)
                logger.info(f"Wrote CSV: {csv_out} ({rows_written} rows)")

                save_processed_entry(
                    log_csv_path, product_id, entry_name, collection_id, "ok", rows_written, os.path.basename(csv_out)
                )

            except Exception as e:
                logger.error(f"Processing error for {entry_name}: {e}")
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "process_error", 0, "")
            finally:
                _cleanup_files(downloaded_files)

            # Timing for this scene
            spent = time.time() - start_t
            total_time += spent
            _append_time(time_log_file, entry_name, spent)
            logger.info(f"Processing time for {entry_name}: {spent:.2f}s")

    logger.info(f"Total processing time: {total_time:.2f}s")

def _cleanup_files(paths: List[str]) -> None:
    """Delete temporary files if they exist; ignore errors."""
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def _append_time(path: str, entry_name: str, seconds: float) -> None:
    """Append a single timing row to time_spent.csv."""
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([entry_name, f"{seconds:.2f}"])

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chlorophyll satellite data downloader/processor with idempotent logging (NT > STC > NR).")
    parser.add_argument("--start", default="2025-06-13", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2025-09-01", help="End date YYYY-MM-DD")

    parser.add_argument("--lon", type=float, default=-117.31646, help="ROI center longitude")
    parser.add_argument("--lat", type=float, default=32.92993,   help="ROI center latitude")
    parser.add_argument("--radius-km", type=float, default=5.0,  help="ROI half-size in kilometers (square ROI)")

    parser.add_argument("--collections", nargs="+", default=["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"],
                        help="EUMETSAT collection IDs to search")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory for CSVs and helper files")

    parser.add_argument("--products", type=str, default="",
                        help="Extra comma-separated entry tokens to include (added to defaults). Case-insensitive, extension-agnostic.")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for logs")
    parser.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help="Processed-log CSV filename (inside --log-dir)")
    parser.add_argument("--file-log", action="store_true", help="Also write a runtime .log file in --log-dir")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Console log level")

    parser.add_argument("--force", action="store_true", help="Reprocess even if product_id is already in the processed log")
    parser.add_argument("--throttle", type=float, default=0.2, help="Pause between entry downloads (seconds)")

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Optional runtime file logger
    ensure_dir(args.log_dir)
    log_csv_path = os.path.join(args.log_dir, args.log_csv)
    if args.file_log:
        fpath = add_file_logger(args.log_dir, level=getattr(logging, args.log_level))
        logger.info(f"Runtime log file: {fpath}")

    # EUMDAC credentials (~/.eumdac/credentials: 'user,pass')
    cred_file = Path.home() / ".eumdac" / "credentials"
    try:
        user, pwd = cred_file.read_text().split(",")
        token = eumdac.AccessToken((user.strip(), pwd.strip()))
        logger.info(f"Token obtained. Expires: {token.expiration}")
    except Exception as e:
        logger.error(f"Error loading EUMDAC credentials: {e}")
        return

    datastore = eumdac.DataStore(token)

    # Extra product tokens from CLI (lowercased)
    extra = normalize_tokens(args.products)

    process_chlorophyll_data(
        datastore=datastore,
        lon=args.lon,
        lat=args.lat,
        radius_km=args.radius_km,
        start_date=args.start,
        end_date=args.end,
        collection_ids=args.collections,
        output_dir=args.out,
        selected_products=extra,
        log_csv_path=log_csv_path,
        force=args.force,
        throttle_s=args.throttle
    )

if __name__ == "__main__":
    main()
