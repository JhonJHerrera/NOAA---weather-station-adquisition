import os
import time
import requests
import pandas as pd
from datetime import datetime


def load_ncei_token(token_file=".ncei_token"):
    """
    Loads the NOAA API token from a local file.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, token_file)
        with open(path, "r") as f:
            token = f.read().strip()
            return token
    except FileNotFoundError:
        raise RuntimeError(f"Token file '{token_file}' not found in {script_dir}.")


def download_ncei_data(
    api_token,
    start_date,
    end_date,
    station_id="GHCND:USW00022521",
    datasetid="GHCND",
    save_csv=True,
    output_dir="data/ncei_data",
    max_retries=5
):
    """
    Downloads daily weather data from NOAA NCEI.

    Parameters:
        - api_token (str): NOAA API token
        - start_date, end_date (str): 'YYYY-MM-DD'
        - station_id (str): NCEI station ID (default: San Juan, PR)
        - datasetid (str): NCEI dataset (default: GHCND)
        - save_csv (bool): Save to CSV
        - output_dir (str): Folder for CSV output
        - max_retries (int): Retry attempts for 503 errors
    """
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": api_token}
    all_data = []
    offset = 1

    while True:
        params = {
            "datasetid": datasetid,
            "datatypeid": ["TMAX", "TMIN", "PRCP", "AWND", "WSF2"],
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "limit": 1000,
            "units": "metric",
            "includemetadata": "false",
            "offset": offset
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if response.status_code == 503 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"503 error received. Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(f"Request failed: {e}")
                    return None

        results = response.json().get("results", [])
        if not results:
            break

        all_data.extend(results)
        offset += 1000

    if not all_data:
        print("No data found.")
        return None

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.pivot_table(index="date", columns="datatype", values="value").reset_index()

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"ncei_{station_id.replace(':', '_')}_{start_date}_to_{end_date}.csv")
        df.to_csv(filename, index=False)
        print(f"Data saved to: {filename}")

    return df


def download_ncei_data_by_year(
    start_date,
    end_date,
    station_id="GHCND:USW00022521", #Luis Muñoz Marín (San Juan International Airport - TJSJ
    datasetid="GHCND",
    token_file=".ncei_token",
    save_csv=True,
    output_dir="data/ncei_data"
):
    """
    Downloads weather data year-by-year and combines it into a single DataFrame.

    Parameters:
        - start_date, end_date (str): 'YYYY-MM-DD'
        - station_id (str): Station ID
        - datasetid (str): Dataset (default: GHCND)
        - token_file (str): Path to token file
        - save_csv (bool): Save yearly and final combined CSV
        - output_dir (str): Directory to save files
    """
    api_token = load_ncei_token(token_file)
    all_years_data = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year, end_year + 1):
        s_date = f"{year}-01-01"
        e_date = f"{year}-12-31"
        if year == end_year:
            e_date = end_date

        print(f"Downloading data: {s_date} to {e_date}")
        try:
            yearly_data = download_ncei_data(
                api_token=api_token,
                start_date=s_date,
                end_date=e_date,
                station_id=station_id,
                datasetid=datasetid,
                save_csv=True,
                output_dir=output_dir
            )

            if yearly_data is not None and not yearly_data.empty:
                all_years_data.append(yearly_data)
            else:
                print(f"No data for year {year}")
        except Exception as e:
            print(f"Error downloading data for {year}: {e}")

        time.sleep(1)

    if not all_years_data:
        print("No data downloaded.")
        return None

    final_df = pd.concat(all_years_data, ignore_index=True).sort_values(by="date")

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        combined_filename = os.path.join(output_dir, f"ncei_{station_id.replace(':', '_')}_{start_date}_to_{end_date}.csv")
        final_df.to_csv(combined_filename, index=False)
        print(f"Combined data saved to: {combined_filename}")

    return final_df


if __name__ == "__main__":
    # Example usage
    download_ncei_data_by_year(
        start_date="2016-01-01",
        end_date="2025-02-01",
        station_id="GHCND:USW00022521",  # San Juan, PR
        datasetid="GHCND",
        token_file=".ncei_token",
        save_csv=True,
        output_dir="data/ncei_data"
    )
