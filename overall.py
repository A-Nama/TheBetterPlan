# streamlit_app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.cm as cm

st.set_page_config(layout="wide", page_title="Urban Planner AI")

st.title("🌍 Interactive Urban Planner (Hackathon Demo)")
st.markdown("Toggle variables on the map and click a location to get AI planning recommendations.")

# -----------------------
# Sidebar: variable selection
# -----------------------
variables = {
    "Temperature (°C)": "data/temperature.tif",
    "Precipitation (mm)": "data/precipitation.tif",
    "Vegetation (Leaf Area Index)": "data/vegetation.tif"
}

selected_var = st.sidebar.selectbox("Select variable to display", list(variables.keys()))

# -----------------------
# Load raster
# -----------------------
raster_path = variables[selected_var]
with rasterio.open(raster_path) as src:
    arr = src.read(1)  # single band
    bounds = src.bounds
    lons = np.linspace(bounds.left, bounds.right, arr.shape[1])
    lats = np.linspace(bounds.bottom, bounds.top, arr.shape[0])
    arr_masked = np.ma.masked_invalid(arr)

# Normalize for color map
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
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# Overlay raster as ImageOverlay
folium.raster_layers.ImageOverlay(
    image=rgba_img,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=0.7,
    interactive=True,
    cross_origin=False,
    zindex=1
).add_to(m)

st.subheader("Click a location on the map")
map_out = st_folium(m, width=800, height=500)

# -----------------------
# On click: show AI recommendations
# -----------------------
if map_out and map_out.get("last_clicked"):
    click = map_out["last_clicked"]
    lat_click, lon_click = click["lat"], click["lng"]
    st.write(f"📍 Selected location: lat={lat_click:.4f}, lon={lon_click:.4f}")

    # Sample raster at click
    row = int((lat_click - bounds.bottom) / (bounds.top - bounds.bottom) * arr.shape[0])
    col = int((lon_click - bounds.left) / (bounds.right - bounds.left) * arr.shape[1])
    row = np.clip(row, 0, arr.shape[0]-1)
    col = np.clip(col, 0, arr.shape[1]-1)
    value = arr[row, col]
    st.write(f"**{selected_var} value at location:** {value:.2f}")

    # Fake AI summary (replace with Gemini/OpenAI)
    summary = {
        "temperature": float(value) if "Temp" in selected_var else 25.0,
        "precipitation": 5.0,
        "vegetation": 0.3
    }

    # Simple rule-based AI (hackathon-ready)
    labels = []
    interventions = []

    if summary["temperature"] >= 35:
        labels.append("Urban Heat Island")
        interventions.append("Plant pocket parks and green roofs.")
    if summary["precipitation"] < 1:
        labels.append("Drought Risk")
        interventions.append("Improve water storage and irrigation.")
    if summary["vegetation"] < 0.2:
        labels.append("Low Vegetation")
        interventions.append("Increase urban greenery and tree planting.")

    if not labels:
        labels.append("No acute issues detected")
        interventions.append("Routine monitoring recommended")

    st.subheader("🏙️ AI Urban Planning Recommendations")
    st.write("**Labels:**", labels)
    st.write("**Interventions:**")
    for i, it in enumerate(interventions, 1):
        st.write(f"{i}. {it}")
