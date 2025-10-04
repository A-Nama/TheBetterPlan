import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "ensemble_mean",
    "variable": [
        "2m_temperature",
        "mean_surface_runoff_rate",
        "mean_total_precipitation_rate",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation"
    ],
    "year": "2019",
    "month": [
        "01", "04", "07",
        "10"
    ],
    "day": [
        "01", "10", "20",
        "30"
    ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",
    "frequency": "6_hourly",
    "area": [45, 122, 24, 153]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
