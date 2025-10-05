import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.errors import RasterioIOError
import numpy as np
import matplotlib.cm as cm
import xarray as xr
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
st.set_page_config(layout="wide", page_title="The Better Plan 🌏")

# Configure Gemini AI Model
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        model = None
        st.warning(f"Could not initialize Gemini model: {e}", icon="⚠️")
else:
    model = None
    st.info("To enable the 'Ask the Planner AI' feature, please add your GOOGLE_API_KEY to a `.env` file.", icon="💡")


# --- UI TITLE ---
st.title("The Better Plan 🏙️")
st.markdown("Click on a region to get urban planning recommendations based on population & environmental data.")


# --- DATASET PATHS ---
DATA_DIR = "data"
COPERNICUS_DIR = os.path.join(DATA_DIR, "copernicus")
NASA_DIR = os.path.join(DATA_DIR, "nasa")

os.makedirs(COPERNICUS_DIR, exist_ok=True)
os.makedirs(NASA_DIR, exist_ok=True)

datasets = {
    # Copernicus Datasets
    "Population Density (people/km²)": os.path.join(COPERNICUS_DIR, "jpn_popden_2019_1km.tif"),
    "Temperature (°C)": os.path.join(COPERNICUS_DIR, "2m_temperature_stream-enda_daily-mean.nc"),
    "Precipitation (mm)": os.path.join(COPERNICUS_DIR, "mean_total_precipitation_rate_2_daily-mean.nc"),
    "Vegetation (Leaf Area Index)": os.path.join(COPERNICUS_DIR, "leaf_area_index_high_vegetation_2_daily-mean.nc"),
    "Runoff Rate": os.path.join(COPERNICUS_DIR, "mean_surface_runoff_rate_2_daily-mean.nc"),
    
    # NASA Dataset
    "Access to Electricity (%)": os.path.join(NASA_DIR, "sdgi_7_1_1_electricity_access.tif"),
}

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(path):
    """Loads raster or NetCDF data from a given path."""
    if not os.path.exists(path):
        st.warning(f"Data file not found: {path}")
        return None, None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".tif", ".tiff"]:
            with rasterio.open(path) as src:
                return src.read(1), src.bounds
        elif ext in [".nc"]:
            with xr.open_dataset(path) as ds:
                varname = list(ds.data_vars)[0]
                da = ds[varname]
                time_dims = [dim for dim in da.dims if "time" in dim.lower()]
                if time_dims:
                    da = da.isel({time_dims[0]: 0})
                da = da.squeeze()
                arr = da.values.astype(np.float32)
                lats, lons = da["latitude"].values, da["longitude"].values
                bounds = rasterio.coords.BoundingBox(lons.min(), lats.min(), lons.max(), lats.max())
                return arr, bounds
    except Exception as e:
        st.error(f"Error loading {os.path.basename(path)}: {e}")
        return None, None
    return None, None

# --- SIDEBAR & MAP DISPLAY ---
st.sidebar.header("Map Settings")
available_vars = [name for name, path in datasets.items() if os.path.exists(path)]
if not available_vars:
    st.error("No data files found! Please place data in 'data/copernicus' and 'data/nasa' folders.")
    st.stop()
    
selected_var = st.sidebar.selectbox("Select variable to display", available_vars)
raster_path = datasets[selected_var]
arr, bounds = load_data(raster_path)

if arr is None:
    st.stop()

arr_masked = np.ma.masked_where((np.isnan(arr)) | (arr <= -999), arr)

with np.errstate(invalid='ignore'):
    writable_masked_data = arr_masked.compressed().copy()
    if "Electricity" in selected_var:
        vmin, vmax = 95, 100
    elif "Population" in selected_var:
        vmin, vmax = 0, np.nanpercentile(writable_masked_data, 99) if writable_masked_data.size > 0 else 1
    else:
        vmin = np.nanpercentile(writable_masked_data, 2) if writable_masked_data.size > 0 else 0
        vmax = np.nanpercentile(writable_masked_data, 98) if writable_masked_data.size > 0 else 1

if vmin is None or vmax is None or vmin == vmax: vmin, vmax = 0, 1

norm = np.clip((arr_masked - vmin) / (vmax - vmin + 1e-9), 0, 1)
cmap = cm.get_cmap("viridis")
rgba_img = (cmap(norm) * 255).astype("uint8")

map_center = [(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2] if bounds else [36.2, 138.25]
m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")
if bounds:
    folium.raster_layers.ImageOverlay(image=rgba_img, bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]], opacity=0.7, interactive=True).add_to(m)

st.subheader("Click on the map to analyze a region")
map_out = st_folium(m, width=800, height=500)

# --- ANALYSIS ON CLICK ---
if map_out and map_out.get("last_clicked"):
    click = map_out["last_clicked"]
    lat, lon = click["lat"], click["lng"]
    st.write(f"📍 **Selected location:** `latitude: {lat:.4f}`, `longitude: {lon:.4f}`")

    def sample_data(path, lat, lon):
        arr_sample, bounds_sample = load_data(path)
        if arr_sample is None: return "N/A"
        h, w = arr_sample.shape
        row = h - int((lat - bounds_sample.bottom) / (bounds_sample.top - bounds_sample.bottom) * h)
        col = int((lon - bounds_sample.left) / (bounds_sample.right - bounds_sample.left) * w)
        val = arr_sample[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]
        return float(val) if np.isfinite(val) and val > -999 else "N/A"

    data_at_click = {name: sample_data(path, lat, lon) for name, path in datasets.items() if os.path.exists(path)}
    st.write("### Data at Selected Location")
    st.table(data_at_click)

    # --- AI ANALYSIS ENGINE ---
    st.subheader("🏙️ Urban Planner AI Recommendations")
    
    temp = data_at_click.get("Temperature (°C)", 0)
    veg_lai = data_at_click.get("Vegetation (Leaf Area Index)", 1)
    pop_den = data_at_click.get("Population Density (people/km²)", 0)
    runoff = data_at_click.get("Runoff Rate", 0)
    elec_access = data_at_click.get("Access to Electricity (%)", 100)

    complex_labels = []
    complex_interventions = []

    # --- COMPLEX SCENARIO ANALYSIS ---
    if isinstance(temp, float) and temp >= 35 and isinstance(veg_lai, float) and veg_lai < 0.2:
        complex_labels.append("🔥 High-Risk Urban Heat Island")
        complex_interventions.append("Critical heat risk due to high temperatures and lack of green cover. **Action:** Prioritize planting urban forests, green roofs/facades, and creating shaded public spaces with water features.")

    if isinstance(pop_den, float) and pop_den > 3000 and isinstance(runoff, float) and runoff > 5:
        complex_labels.append("💧 High-Density Flood Risk Zone")
        complex_interventions.append("This dense area is vulnerable to surface flooding. **Action:** Implement sustainable urban drainage (SUDS) like permeable pavements and bioswales. Protect and expand green spaces to absorb rainwater.")

    if isinstance(pop_den, float) and pop_den > 1000 and isinstance(elec_access, float) and elec_access < 100:
        complex_labels.append("⚡ Energy Inequality Hotspot")
        complex_interventions.append("A populated area with an electricity service gap. **Action:** This is a priority for grid extension or deploying community-scale renewable microgrids to ensure equitable and reliable energy.")

    if complex_labels:
        st.write("#### Complex Scenarios Identified:")
        for i, (label, intervention) in enumerate(zip(complex_labels, complex_interventions), 1):
            st.markdown(f"**{i}. {label}**")
            st.markdown(intervention)
    else:
        st.info("No complex, multi-factor risk scenarios were identified at this location.", icon="✅")
        if model:
            st.write("#### General Improvement Tips:")
            prompt_context = "\n".join([f"- {k}: {v}" for k, v in data_at_click.items()])
            prompt = f"""
            You are an expert urban planner. Based on the following data for a location in Japan, generate 2-3 concise, actionable tips for general urban improvement. Focus on sustainability and resilience.
            
            Data:
            {prompt_context}

            Your response must be a brief, bulleted list. Do not add any introductory or concluding sentences.
            """
            with st.spinner("Generating tips..."):
                try:
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Could not generate tips: {e}")

    # --- GEMINI CONVERSATIONAL AI ---
    if model:
        st.subheader("💬 Ask The Planner AI")
        prompt_context = "\n".join([f"- {k}: {v}" for k,v in data_at_click.items()])
        scenario_context = "\n".join([f"- {l}" for l in complex_labels])

        user_question = st.text_input(
            "Ask a follow-up question:",
            placeholder="e.g., What kind of trees are best for urban cooling here?"
        )

        if user_question:
            # --- MODIFIED PROMPT FOR CONCISENESS ---
            prompt = f"""
            You are an expert urban planner for the NASA Space Apps Challenge, analyzing a location in Japan.

            **Raw Data:**
            {prompt_context}

            **AI-Identified Scenarios:**
            {scenario_context if scenario_context else "No complex scenarios identified."}

            Based on all this information, answer the user's question. 
            **Your response must be very concise, practical, and limited to 2-3 key sentences.**

            User's question: "{user_question}"
            """
            with st.spinner("The Planner AI is thinking..."):
                try:
                    response = model.generate_content(prompt)
                    st.markdown("**AI Response:**")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Could not get a response from the AI model: {e}")

