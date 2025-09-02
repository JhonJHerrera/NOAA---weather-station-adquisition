from datetime import datetime, timedelta
from tides_pipeline import run_multi_month_download, merge_products_daily_avg
from ncei_pipeline import load_ncei_token, download_ncei_data

def run_all_for_yesterday():
    # Calcular la fecha de ayer
    yesterday = datetime.utcnow() - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")

    print(f"\n==== Running full pipeline for {date_str} ====\n")

    # --- TIDES DATA ---
    run_multi_month_download(
        start="2020-01-01",
        end="2025-06-30",
        station_id="9755371",  # La Puntilla, San Juan
        output_dir="data/tide_data"
    )
    merge_products_daily_avg(
        output_dir="data/tide_data",
        save_path="data/combined/tide_data.csv"
    )

    # --- NCEI DATA ---
    api_token = load_ncei_token(token_file=".ncei_token")
    download_ncei_data(
        api_token=api_token,
        start_date="2020-01-01",
        end_date="2025-06-30",
        station_id="GHCND:USW00022521",  # San Juan Intl Airport
        datasetid="GHCND",
        save_csv=True,
        output_dir="data/ncei_data"
    )

if __name__ == "__main__":
    run_all_for_yesterday()
