import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os

def rasterize_electricity_data():
    """
    Converts the NASA SDGI 7.1.1 (Access to Electricity) vector shapefile
    to a GeoTIFF raster, filtering for Japan's subnational data.
    """
    print("Starting rasterization process for Japan (Access to Electricity)...")

    # --- 1. DEFINE FILE PATHS ---
    # Put the unzipped shapefile components in this folder
    input_shapefile_dir = "sdgi-7-1-1-access-electricity-2023-shp"
    
    # --- IMPORTANT: We want the SUBNATIONAL file for a detailed map ---
    target_filename_part = "subnational.shp"
    
    try:
        # Find the specific subnational shapefile
        shp_file = next(f for f in os.listdir(input_shapefile_dir) if f.endswith(target_filename_part))
        input_shapefile_path = os.path.join(input_shapefile_dir, shp_file)
    except (StopIteration, FileNotFoundError):
        print(f"ERROR: No file ending in '{target_filename_part}' found in the '{input_shapefile_dir}' directory.")
        print("ACTION: Please download the 'Access to Electricity' shapefile and unzip it into the 'source_shapefile' folder.")
        return

    # Define a new, descriptive output name
    output_dir = os.path.join("data", "nasa")
    os.makedirs(output_dir, exist_ok=True)
    output_raster_path = os.path.join(output_dir, "sdgi_7_1_1_electricity_access.tif")
    
    print(f"Input shapefile: {input_shapefile_path}")
    print(f"Output raster: {output_raster_path}")

    # --- 2. LOAD THE SHAPEFILE ---
    try:
        gdf = gpd.read_file(input_shapefile_path)
    except Exception as e:
        print(f"ERROR: Could not read the shapefile. Details: {e}")
        return

    # --- 3. CORRECT COLUMN NAMES FOR THIS DATASET ---
    burn_attribute = 'SDG711pct'      # This is the data column for electricity access %
    country_iso_column = 'ISO3'         # This is the country code column
    country_iso_code = 'JPN'            # ISO code for Japan

    if burn_attribute not in gdf.columns:
        print(f"ERROR: The required data column '{burn_attribute}' was not found.")
        print(f"Available columns are: {list(gdf.columns)}")
        return
        
    print("Shapefile loaded successfully.")

    # --- 4. FILTER DATA FOR JAPAN ---
    if country_iso_column not in gdf.columns:
        print(f"ERROR: Country ISO column '{country_iso_column}' not found.")
        return

    print(f"Filtering data for country code: {country_iso_code}...")
    gdf_country = gdf[gdf[country_iso_column] == country_iso_code].copy()

    if gdf_country.empty:
        print(f"ERROR: No data found for country code '{country_iso_code}'.")
        return
    
    print(f"Found {len(gdf_country)} subnational features for Japan.")
    
    # --- 5. DEFINE RASTER PROPERTIES BASED ON JAPAN'S DATA ---
    bounds = gdf_country.total_bounds
    buffer = 0.1
    min_lon, min_lat, max_lon, max_lat = bounds
    bounds = (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
    
    resolution = 0.01

    out_width = int((bounds[2] - bounds[0]) / resolution)
    out_height = int((bounds[3] - bounds[1]) / resolution)
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution, resolution)

    print(f"Output raster dimensions: {out_height} rows x {out_width} columns")
    
    # --- 6. RASTERIZE THE FILTERED DATA ---
    gdf_country = gdf_country.dropna(subset=[burn_attribute])
    gdf_country = gdf_country[gdf_country[burn_attribute] >= 0]
    
    shapes = ((geom, value) for geom, value in zip(gdf_country.geometry, gdf_country[burn_attribute]))
    raster_array = np.full((out_height, out_width), -9999.0, dtype=np.float32)

    print("Burning Japan's vector features onto the raster grid...")
    features.rasterize(
        shapes=shapes,
        out=raster_array,
        transform=transform,
        fill=-9999.0,
        all_touched=True,
        dtype=np.float32
    )

    # --- 7. SAVE THE RASTER TO A .TIF FILE ---
    profile = {
        'driver': 'GTiff', 'height': out_height, 'width': out_width, 'count': 1,
        'dtype': np.float32, 'crs': gdf_country.crs, 'transform': transform, 'nodata': -9999.0
    }

    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(raster_array, 1)

    print("-" * 30)
    print("✅ Rasterization for Japan (Electricity Access) complete!")
    print(f"GeoTIFF file saved to: {output_raster_path}")
    print("-" * 30)

if __name__ == "__main__":
    rasterize_electricity_data()

