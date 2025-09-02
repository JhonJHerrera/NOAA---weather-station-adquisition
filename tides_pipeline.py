import os
import time
import calendar
import requests
import pandas as pd
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY

def download_tide_data(station_id, start_date, end_date, product, interval='h', max_retries=3):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "station": station_id,
        "begin_date": start_date,
        "end_date": end_date,
        "product": product,
        "datum": "MLLW",
        "interval": interval,
        "units": "metric",
        "time_zone": "gmt",
        "format": "json"
    }

    print(f"Fetching '{product}' from {start_date} to {end_date} for station {station_id}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                print(f"Error for product '{product}': {data['error']['message']}")
                return None
            key = "data" if "data" in data else "predictions"
            df = pd.DataFrame(data[key])
            df["t"] = pd.to_datetime(df["t"])
            df["v"] = pd.to_numeric(df["v"], errors="coerce")
            return df
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                print(f"400 error for product '{product}': {e}")
                return None
            elif attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Request error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed to download product '{product}' after {max_retries} attempts.")
                return None

def run_multi_month_download(
    start,
    end,
    station_id="9755371",
    products=["air_temperature", "water_temperature", "air_pressure",'water_level'],
    output_dir="data/tide_data"
):
    from glob import glob

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)

    all_product_dfs = {product: [] for product in products}

    for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt):
        year = dt.year
        month = dt.month
        first_day = f"{year}-{month:02d}-01"
        _, last_day = calendar.monthrange(year, month)
        last_day_str = f"{year}-{month:02d}-{last_day:02d}"

        print(f"\nDownloading: {first_day} → {last_day_str}")

        for product in products:
            df = download_tide_data(station_id, first_day, last_day_str, product)
            if df is not None:
                df = df.rename(columns={"t": "date", "v": product})
                df = df[["date", product]]
                all_product_dfs[product].append(df)

    # Guardar CSV por producto
    for product, df_list in all_product_dfs.items():
        if df_list:
            full_df = pd.concat(df_list)
            full_df = full_df.sort_values("date")
            full_path = os.path.join(output_dir, f"{product}_full.csv")
            full_df.to_csv(full_path, index=False)
            print(f"Saved combined {product} data to {full_path}")

def merge_products_daily_avg(output_dir="data/tide_data", save_path="data/combined/tide_data.csv"):
    import glob

    print("\nMerging all product files by 'date' with daily averages...")

    product_files = glob.glob(os.path.join(output_dir, "*_full.csv"))
    merged_df = None

    for file in product_files:
        df = pd.read_csv(file)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.date
        value_col = df.columns[1]  # product name
        daily_avg = df.groupby("day")[value_col].mean().reset_index()
        daily_avg.rename(columns={"day": "date"}, inplace=True)

        if merged_df is None:
            merged_df = daily_avg
        else:
            merged_df = pd.merge(merged_df, daily_avg, on="date", how="outer")

        # Remove temporary product file
        os.remove(file)
        print(f"Deleted temp file: {file}")

    merged_df = merged_df.sort_values("date")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    print(f"Saved daily averaged merged data to {save_path}")

if __name__ == "__main__":
    run_multi_month_download(
        start="2025-01-01",
        end="2025-07-20",
        station_id="9755371",
        output_dir="data/tide_data"
    )
    merge_products_daily_avg()
