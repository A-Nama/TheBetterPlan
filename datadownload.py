import streamlit as st
import cdsapi
import xarray as xr
import numpy as np
import os
import json
from datetime import datetime
from io import BytesIO
from rasterio.enums import Resampling
import rasterio
import geopandas as gpd
from shapely.geometry import box, mapping
from streamlit_folium import st_folium
import folium
import requests
import tempfile
import math

st.set_page_config(layout="wide", page_title="CITI-HEAL Planner (Copernicus + LLM)")

# ---------------------------
# Config & helpers
# ---------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Simple caching helpers (filesystem)
def cache_key_for_request(bbox, date_str, variables):
    return f"era5_{date_str}_{variables}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}".replace(" ", "")

def cached_filepath(key, ext=".nc"):
    safe = key.replace(":", "_")
    return os.path.join(DATA_DIR, f"{safe}{ext}")

# ---------------------------
# Download ERA5-Land subset via CDS API (NetCDF).
# ---------------------------
def download_era5_land(bbox, date_str, variables, out_path):
    c = cdsapi.Client()
    # CDS area param order: North, West, South, East
    area = [bbox[3], bbox[0], bbox[1], bbox[2]]
    year, month, day = date_str.split("-")
    req = {
        "variable": variables,
        "year": year,
        "month": month,
        "day": day,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": area,
        "data_format": "netcdf",
        "product_type": "reanalysis"
    }
    st.info("Requesting ERA5-Land subset from CDS (this may take 30s-2min)...")
    try:
        c.retrieve('reanalysis-era5-land', req, out_path)
    except Exception as e:
        st.error(f"CDS download error: {e}")
        raise
    st.success(f"Saved ERA5-Land to {out_path}")
    return out_path

# ---------------------------
# Safe loaders for NetCDF and GeoTIFF (preview / downsample)
# ---------------------------
def open_dataset_safe(nc_path, engine_preference=["netcdf4","h5netcdf"]):
    last_err = None
    for eng in engine_preference:
        try:
            ds = xr.open_dataset(nc_path, engine=eng, decode_times=True, mask_and_scale=True)
            return ds
        except Exception as e:
            last_err = e
    # fallback: try no engine (let xarray pick)
    try:
        ds = xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True)
        return ds
    except Exception as e:
        raise last_err or e

def coarsen_da(da, max_pixels=800*800):
    da = da.squeeze()
    if da.ndim != 2:
        # try to reduce extra dims
        try:
            da = da.isel({d:0 for d in da.dims if d not in ('latitude','longitude','lat','lon','y','x')})
        except Exception:
            da = da.isel({da.dims[0]:0})
    ny, nx = da.shape[-2], da.shape[-1]
    n_pixels = int(ny) * int(nx)
    if n_pixels > max_pixels:
        factor = int(np.ceil((n_pixels / max_pixels) ** 0.5))
        dims = {da.dims[-2]: factor, da.dims[-1]: factor}
        try:
            da = da.coarsen(**dims, boundary="trim").mean()
        except Exception:
            da = da[::factor, ::factor]
    return da

def load_and_coarsen_nc(nc_path, varname, time_index=0, max_pixels=800*800):
    ds = open_dataset_safe(nc_path)
    if varname not in ds.data_vars:
        raise ValueError(f"{varname} not in dataset variables: {list(ds.data_vars.keys())}")
    da = ds[varname]
    # select time if present
    if "time" in da.dims:
        da = da.isel(time=time_index)
    da = coarsen_da(da, max_pixels=max_pixels)
    arr = da.values.astype("float32")
    # coords names
    lat_name = next((n for n in ("latitude","lat","y") if n in ds.coords), None)
    lon_name = next((n for n in ("longitude","lon","x") if n in ds.coords), None)
    if lat_name and lon_name:
        lats = ds[lat_name].values
        lons = ds[lon_name].values
    else:
        # generate approximate coords using bbox if available in attrs
        lats = np.linspace(float(getattr(ds, "south_north", -90)), float(getattr(ds, "north_south", 90)), arr.shape[0])
        lons = np.linspace(float(getattr(ds, "west_east", -180)), float(getattr(ds, "east_west", 180)), arr.shape[1])
    return arr, lons, lats, getattr(da, "units", "")

def read_tif_preview(tif_path, out_shape=(512,512)):
    # accepts path or file-like
    with rasterio.open(tif_path) as src:
        # compute target shape keeping aspect
        height, width = src.height, src.width
        # if already small, read native
        if height * width <= out_shape[0] * out_shape[1]:
            band = src.read(1, masked=True).astype("float32")
            transform = src.transform
        else:
            # compute scale for out_shape approximate
            scale_x = width / out_shape[1]
            scale_y = height / out_shape[0]
            out_w = out_shape[1]
            out_h = out_shape[0]
            band = src.read(1, out_shape=(out_h,out_w), resampling=Resampling.average).astype("float32")
            transform = src.transform  # not exact for preview but fine
        # get bounds
        bounds = src.bounds
    # build coords (lon, lat arrays)
    lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
    lons = np.linspace(lon_min, lon_max, band.shape[1])
    lats = np.linspace(lat_min, lat_max, band.shape[0])
    return band, lons, lats

# ---------------------------
# Geo helpers
# ---------------------------
def bbox_from_center(lat, lon, half_deg):
    return (lon-half_deg, lat-half_deg, lon+half_deg, lat+half_deg)

# ---------------------------
# Agent (rule-based fallback)
# ---------------------------
def rule_based_agent(summary):
    labels = []
    interventions = []
    if summary.get("mean_celsius", -999) >= 35 and summary.get("pop_density",0) >= 3000:
        labels.append("Urban Heat Island")
        interventions.append("Increase canopy cover & create pocket parks.")
    if summary.get("pm25", 0) >= 35:
        labels.append("Pollution Hotspot")
        interventions.append("Add green buffers and target industrial emission controls.")
    if summary.get("healthcare_distance_min", 999) >= 0.2:  # degrees approx => ~20-30 km scaled; interpret accordingly
        labels.append("Healthcare Desert")
        interventions.append("Establish a primary health clinic or mobile health unit.")
    if summary.get("energy_access", 1) <= 0.35:
        labels.append("Energy-Poor Zone")
        interventions.append("Pilot solar microgrid + cold storage.")
    if not labels:
        labels.append("No acute condition detected; moderate monitoring recommended.")
        interventions.append("Routine monitoring + community engagement.")
    diagnosis = " / ".join(labels) + ". " + " ".join(interventions[:2])
    return {"labels": labels, "diagnosis": diagnosis, "interventions": interventions}

def call_llm_agent(summary_json):
    api_key = os.environ.get("LLM_API_KEY", None)
    if not api_key:
        return rule_based_agent(summary_json)
    # Put provider-specific code here. For hackathon, fallback to rule-based.
    return rule_based_agent(summary_json)

# ---------------------------
# App UI
# ---------------------------
st.title("CITI-HEAL Planner — Copernicus (ERA5-Land) + LLM")
st.markdown("Select place & date → run analysis (CDS or upload) → Ask the AI planner for recommendations.")

sidebar = st.sidebar
sidebar.header("Area & date")
lat = sidebar.number_input("Center latitude", value=35.68, format="%.6f")
lon = sidebar.number_input("Center longitude", value=139.76, format="%.6f")
half_deg = sidebar.slider("bbox half-width (deg)", 0.02, 1.0, 0.2, 0.02)
date_selected = sidebar.date_input("Date", value=datetime(2024,1,1))
date_str = date_selected.strftime("%Y-%m-%d")

variables = sidebar.multiselect("Variables to request (ERA5-Land)",
                                ["2m_temperature", "total_precipitation",
                                 "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"],
                                default=["2m_temperature"])
run_download = sidebar.button("Run analysis (download & compute)")

# Upload local files (NetCDF or GeoTIFF) — if used, download step is skipped
st.sidebar.markdown("Or upload local dataset (NetCDF .nc or GeoTIFF .tif)")
uploaded_nc = st.sidebar.file_uploader("Upload NetCDF (.nc)", type=["nc"])
uploaded_tif = st.sidebar.file_uploader("Upload GeoTIFF (.tif,.tiff)", type=["tif","tiff"])

# optional population GeoTIFF
worldpop_path = sidebar.file_uploader("Optional: WorldPop GeoTIFF (for pop density)", type=["tif","tiff"])

# interactive map for picking area visually
st.subheader("Map (click to set bbox center)")
m = folium.Map(location=[lat, lon], zoom_start=9, tiles="CartoDB positron")
folium.Rectangle(bounds=[(lat-half_deg, lon-half_deg),(lat+half_deg, lon+half_deg)], color="blue").add_to(m)
map_out = st_folium(m, width=700, height=400)
if map_out and map_out.get("last_clicked"):
    click = map_out["last_clicked"]
    lat, lon = click["lat"], click["lng"]

bbox = bbox_from_center(lat, lon, half_deg)
key = cache_key_for_request(bbox, date_str, ",".join(variables))
nc_cache = cached_filepath(key, ext=".nc")

# If user uploaded a NetCDF, save to cache path and use it
if uploaded_nc:
    tmp_path = os.path.join(DATA_DIR, uploaded_nc.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_nc.read())
    nc_cache = tmp_path
    st.sidebar.success(f"Using uploaded NetCDF: {uploaded_nc.name}")

# If user uploaded a GeoTIFF, save path and set mode to raster preview
tif_cache = None
if uploaded_tif:
    tmp_tif = os.path.join(DATA_DIR, uploaded_tif.name)
    with open(tmp_tif, "wb") as f:
        f.write(uploaded_tif.read())
    tif_cache = tmp_tif
    st.sidebar.success(f"Using uploaded GeoTIFF: {uploaded_tif.name}")

# If user clicked Run Analysis, download ERA5 unless nc_cache already exists
if run_download and not os.path.exists(nc_cache):
    try:
        tmp = nc_cache + ".tmp"
        download_era5_land(bbox, date_str, variables, tmp)
        os.replace(tmp, nc_cache)
    except Exception as e:
        st.error("Download failed: " + str(e))
        st.stop()

# Decide which data source to preview: priority: uploaded_tif -> uploaded_nc -> cached nc (downloaded)
data_source_type = None
display_units = ""
display_arr = None
lons = lats = None

if tif_cache:
    try:
        band, lons, lats = read_tif_preview(tif_cache, out_shape=(512,512))
        display_arr = band
        display_units = ""
        data_source_type = "tif"
    except Exception as e:
        st.error("Failed to load uploaded GeoTIFF: " + str(e))
        st.stop()
elif os.path.exists(nc_cache):
    # preview NetCDF: let user choose variable/time
    try:
        # open dataset to list vars & times
        ds = open_dataset_safe(nc_cache)
        vars_list = list(ds.data_vars.keys())
        display_var = st.selectbox("Variable to preview (from NetCDF)", vars_list)
        # if time present, show times
        da = ds[display_var]
        time_dim = next((d for d in da.dims if "time" in d.lower()), None)
        time_index = 0
        if time_dim:
            times = ds[time_dim].values
            sel = st.selectbox("Select time", [str(t) for t in times], index=0)
            time_index = [str(t) for t in times].index(sel)
        arr, lons, lats, units = load_and_coarsen_nc(nc_cache, display_var, time_index=time_index, max_pixels=600*600)
        display_arr = arr
        display_units = units
        data_source_type = "nc"
    except Exception as e:
        st.error("Failed to load NetCDF preview: " + str(e))
        st.stop()
else:
    st.info("No data selected yet. Upload a file or Run analysis to download ERA5 subset.")

# If we have a display array, render map and metrics
if display_arr is not None:
    st.subheader("Preview map & metrics")
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    m2 = folium.Map(location=[(lat_min+lat_max)/2, (lon_min+lon_max)/2], zoom_start=9, tiles="CartoDB positron")
    # normalize for overlay
    vmin = float(np.nanpercentile(display_arr, 2))
    vmax = float(np.nanpercentile(display_arr, 98))
    norm = (display_arr - vmin) / (vmax - vmin + 1e-9)
    norm = np.clip(norm, 0, 1)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("inferno")
    rgba_img = (cmap(norm) * 255).astype("uint8")
    folium.raster_layers.ImageOverlay(
        image=rgba_img,
        bounds=bounds,
        opacity=0.75,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m2)
    # draw chosen bbox
    folium.Rectangle(bounds=[[bbox[1], bbox[0]],[bbox[3], bbox[2]]], color="lime").add_to(m2)
    st_data = st_folium(m2, width=700, height=450)

    # metrics
    # convert temperature if variable suggests temperature (netcdf var name contains 'temp' or 't2m')
    mean_v = float(np.nanmean(display_arr))
    median_v = float(np.nanmedian(display_arr))
    min_v = float(np.nanmin(display_arr))
    max_v = float(np.nanmax(display_arr))
    if data_source_type == "nc" and ("temp" in display_var.lower() or "t2m" in display_var.lower() or "temperature" in display_var.lower()):
        mean_c = mean_v - 273.15
        median_c = median_v - 273.15
        min_c = min_v - 273.15
        max_c = max_v - 273.15
        display_units = "°C"
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{mean_c:.2f} {display_units}")
        col2.metric("Median", f"{median_c:.2f} {display_units}")
        col3.metric("Min / Max", f"{min_c:.2f} / {max_c:.2f} {display_units}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{mean_v:.2f} {display_units}")
        col2.metric("Median", f"{median_v:.2f} {display_units}")
        col3.metric("Min / Max", f"{min_v:.2f} / {max_v:.2f} {display_units}")

    # population-weighted exposure (optional)
    pop_stats = {}
    if worldpop_path:
        if hasattr(worldpop_path, "read"):
            tmp_pop = os.path.join(DATA_DIR, "worldpop_tmp.tif")
            with open(tmp_pop, "wb") as f:
                f.write(worldpop_path.read())
            pop_file = tmp_pop
        else:
            pop_file = worldpop_path
        try:
            with rasterio.open(pop_file) as src:
                xs = np.linspace(lon_min, lon_max, display_arr.shape[1])
                ys = np.linspace(lat_min, lat_max, display_arr.shape[0])
                pts = [(x,y) for y in ys for x in xs]
                samples = list(src.sample(pts))
                pop_arr = np.array([s[0] if s is not None else 0 for s in samples]).reshape(display_arr.shape)
                pop_stats["pop_sum"] = int(np.nansum(pop_arr))
                pop_stats["pop_mean"] = float(np.nanmean(pop_arr))
                pw_mean = float(np.nansum((display_arr * pop_arr)) / (np.nansum(pop_arr)+1e-9))
                pop_stats["pop_weighted_mean"] = pw_mean
                st.write("Population weighted mean:", pw_mean)
        except Exception as e:
            st.warning("Population sampling failed: " + str(e))

    # healthcare accessibility mock (if facilities.geojson exists)
    facilities_file = os.path.join(DATA_DIR, "facilities.geojson")
    healthcare_stats = {}
    if os.path.exists(facilities_file):
        try:
            gdf = gpd.read_file(facilities_file)
            centroid = ((lon_min+lon_max)/2, (lat_min+lat_max)/2)
            dists = gdf.geometry.apply(lambda p: math.hypot(p.x - centroid[0], p.y - centroid[1]))
            healthcare_stats["min_distance_deg"] = float(dists.min())
            st.write("Healthcare min distance (deg):", healthcare_stats["min_distance_deg"])
        except Exception as e:
            st.warning("Failed to compute healthcare stats: " + str(e))

    # assemble summary
    summary = {
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "date": date_str,
        "data_source": data_source_type,
        "variable": display_var if data_source_type=="nc" else os.path.basename(tif_cache) if tif_cache else "uploaded_tif",
        "mean": mean_v,
        "median": median_v,
        "min": min_v,
        "max": max_v,
        "units": display_units
    }
    summary.update(pop_stats)
    summary.update(healthcare_stats)
    st.write("### Quick summary (JSON)")
    st.json(summary)

    # Ask the planner
    if st.button("Ask the AI planner for recommendations"):
        with st.spinner("Running planner agent..."):
            agent_out = call_llm_agent(summary)
        st.success("Planner recommendations ready")
        st.write("### Labels")
        st.write(agent_out.get("labels", []))
        st.write("### Diagnosis")
        st.write(agent_out.get("diagnosis", ""))
        st.write("### Interventions")
        for i, it in enumerate(agent_out.get("interventions", []), 1):
            st.write(f"{i}. {it}")
        # cache agent output
        cache_agent_file = cached_filepath(key + "_agent", ext=".json")
        with open(cache_agent_file, "w") as f:
            json.dump({"summary": summary, "agent": agent_out}, f, indent=2)
        st.info(f"Saved agent output to {cache_agent_file}")

else:
    st.info("Upload a GeoTIFF/NetCDF or run 'Run analysis' to download ERA5 subset and preview.")
