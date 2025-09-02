https://tidesandcurrents.noaa.gov/stationhome.html?id=9755371

precipitation 
https://www.weather.gov/wrh/Climate?wfo=sju

README - Weather and Tide Data

This dataset combines daily weather observations from the NOAA NCEI station at San Juan Airport (TJSJ)
and tide/coastal conditions from the NOAA CO-OPS station at La Puntilla (9755371), San Juan Bay.

Variables:

From NCEI (TJSJ):
- TMAX: Daily maximum temperature (°C × 0.1)
- TMIN: Daily minimum temperature (°C × 0.1)
- PRCP: Daily precipitation (mm × 0.1)
- AWND: Average wind speed (m/s × 0.1)
- WSF2: Fastest 2-minute wind (m/s × 0.1)

From CO-OPS (La Puntilla):
- water_level: Mean daily water level (meters)
- air_temperature: Mean daily air temperature (°C)
- water_temperature: Mean daily water temperature (°C)
- air_pressure: Mean daily atmospheric pressure (hPa)

Missing or unavailable products per station have been skipped automatically.

All data is merged by date and saved in: combined_weather_tide.csv