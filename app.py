# streamlit_urban_planner.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.cm as cm
import xarray as xr
import os
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found! Please check your .env file.")
else:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")




st.set_page_config(layout="wide", page_title="The Better Plan 🌏")
st.title("The Better Plan 🏙️")
st.markdown("Click on a region to get urban planning recommendations based on population & environmental data.")

# -----------------------
# Dataset paths
# -----------------------
DATA_DIR = "data\copernicus"

datasets = {
    "Population Density (people/km²)": os.path.join(DATA_DIR, "jpn_popden_2019_1km.tif"),
    "Temperature (°C)": os.path.join(DATA_DIR, "2m_temperature_stream-enda_daily-mean.nc"),
    "Precipitation (mm)": os.path.join(DATA_DIR, "mean_total_precipitation_rate_2_daily-mean.nc"),
    "Vegetation (Leaf Area Index)": os.path.join(DATA_DIR, "leaf_area_index_high_vegetation_2_daily-mean.nc"),
    "Runoff Rate": os.path.join(DATA_DIR, "mean_surface_runoff_rate_2_daily-mean.nc"),
    "Low Veg LAI": os.path.join(DATA_DIR, "leaf_area_index_low_vegetation_2_daily-mean.nc"),
}


# -----------------------
# Helper: load raster or netcdf dynamically
# -----------------------
def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with rasterio.open(path) as src:
            arr = src.read(1)
            bounds = src.bounds
        return arr, bounds
    elif ext in [".nc"]:
        ds = xr.open_dataset(path)
        # pick first time dimension dynamically
        varname = list(ds.data_vars)[0]  # pick first variable
        da = ds[varname]
        time_dims = ["time", "valid_time", "forecast_time"]
        for td in time_dims:
            if td in da.dims:
                da = da.isel({td: 0})
                break
        da = da.squeeze()
        arr = da.values.astype(np.float32)
        # bounds
        lats = ds["latitude"].values if "latitude" in ds.coords else np.linspace(-90, 90, arr.shape[0])
        lons = ds["longitude"].values if "longitude" in ds.coords else np.linspace(-180, 180, arr.shape[1])
        bounds = rasterio.coords.BoundingBox(left=float(lons.min()), bottom=float(lats.min()),
                                             right=float(lons.max()), top=float(lats.max()))
        return arr, bounds
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -----------------------
# Variable selection
# -----------------------
selected_var = st.sidebar.selectbox("Select variable to display on map", list(datasets.keys()))
raster_path = datasets[selected_var]

# load data
arr, bounds = load_data(raster_path)
arr_masked = np.ma.masked_invalid(arr)

# normalize for colormap
if "Population Density" in selected_var:
    vmin, vmax = 0, np.nanpercentile(arr, 99)
else:
    vmin = np.nanpercentile(arr, 2)
    vmax = np.nanpercentile(arr, 98)

norm = (arr_masked - vmin) / (vmax - vmin + 1e-9)
norm = np.clip(norm, 0, 1)
cmap = cm.get_cmap("inferno")
rgba_img = (cmap(norm) * 255).astype("uint8")

# -----------------------
# Folium map
# -----------------------
center_lat = (bounds.top + bounds.bottom) / 2
center_lon = (bounds.left + bounds.right) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")
folium.raster_layers.ImageOverlay(
    image=rgba_img,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=0.7,
    interactive=True,
    cross_origin=False,
    zindex=1
).add_to(m)

st.subheader("Click on the map to analyze a region")
map_out = st_folium(m, width=800, height=500)

# -----------------------
# When user clicks
# -----------------------
if map_out and map_out.get("last_clicked"):
    click = map_out["last_clicked"]
    lat_click, lon_click = click["lat"], click["lng"]
    st.write(f"📍 Selected location: lat={lat_click:.4f}, lon={lon_click:.4f}")

    # Function to sample raster/NetCDF at click
    def sample_data(path, lat, lon):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".tif", ".tiff"]:
            with rasterio.open(path) as src:
                arr = src.read(1)
                bounds = src.bounds
                row = int((lat - bounds.bottom) / (bounds.top - bounds.bottom) * arr.shape[0])
                col = int((lon - bounds.left) / (bounds.right - bounds.left) * arr.shape[1])
                row = np.clip(row, 0, arr.shape[0]-1)
                col = np.clip(col, 0, arr.shape[1]-1)
                return float(arr[row, col])
        elif ext in [".nc"]:
            ds = xr.open_dataset(path)
            varname = list(ds.data_vars)[0]
            da = ds[varname]
            time_dims = ["time", "valid_time", "forecast_time"]
            for td in time_dims:
                if td in da.dims:
                    da = da.isel({td: 0})
                    break
            da = da.squeeze()
            arr = da.values
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            row = int((lat - lats.min()) / (lats.max() - lats.min()) * arr.shape[0])
            col = int((lon - lons.min()) / (lons.max() - lons.min()) * arr.shape[1])
            row = np.clip(row, 0, arr.shape[0]-1)
            col = np.clip(col, 0, arr.shape[1]-1)
            return float(arr[row, col])
        else:
            return np.nan

    # collect data from all datasets
    data_at_click = {name: sample_data(path, lat_click, lon_click) for name, path in datasets.items()}
    st.write("### Data at selected location")
    st.json(data_at_click)

    # -----------------------
    # Simple rule-based AI
    # -----------------------
    labels = []
    interventions = []

    # Population density
    if data_at_click.get("Population Density", 0) > 3000:
        labels.append("High Population Density")
        interventions.append("Consider public transport improvements & high-density housing planning.")

    # Temperature
    if data_at_click.get("Temperature (°C)", 0) >= 35:
        labels.append("Urban Heat Island")
        interventions.append("Increase green cover, shade, and reflective surfaces.")

    # Precipitation
    if data_at_click.get("Precipitation (mm)", 0) < 1:
        labels.append("Drought Risk")
        interventions.append("Water harvesting & drought-resilient urban design.")

    # Vegetation
    if data_at_click.get("Vegetation (Leaf Area Index)", 0) < 0.2:
        labels.append("Low Vegetation")
        interventions.append("Plant urban trees & pocket parks.")

    if not labels:
        labels.append("Moderate/No acute risks")
        interventions.append("Routine monitoring & incremental planning.")

    st.subheader("🏙️ Urban Planner AI Recommendations")
    st.write("**Labels:**", labels)
    st.write("**Interventions:**")
    for i, it in enumerate(interventions, 1):
        st.write(f"{i}. {it}")

        # -----------------------
    # Gemini-based conversational AI
    # -----------------------
    st.subheader("💬 Ask The Planner AI")

    user_question = st.text_input(
        "Ask a question about this location:",
        placeholder="e.g. What sustainable strategies suit this region?"
    )

    if user_question:
        # Give Gemini context from the datasets
        context = "\n".join([f"{k}: {v}" for k, v in data_at_click.items()])

        prompt = f"""
        You are an expert urban planner and environmental advisor.
        Here’s the environmental context for a selected location:

        {context}

        Labels identified: {labels}
        Interventions suggested: {interventions}

        The user asks: "{user_question}"

        Please provide a practical, location-specific, and sustainability-focused answer.
        Mention potential urban design ideas, green infrastructure, and adaptation measures.
        """

        with st.spinner("The Planner AI is thinking..."):
            response = model.generate_content(prompt)

        st.markdown("**AI Answer:**")
        st.write(response.text)

