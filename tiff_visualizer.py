import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

# Set page config
st.set_page_config(
    page_title="TIFF Data Visualizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_tiff_file(file_path):
    """Load a TIFF file and return its data and metadata"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds, src.meta
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None, None, None, None, None

def get_tiff_files():
    """Get all TIFF files from the datasets folder"""
    tiff_files = glob.glob("datasets/*.tif")
    return tiff_files

def create_colormap(data, colormap_name='viridis'):
    """Create a colormap for the data"""
    if colormap_name == 'custom_blue':
        colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    else:
        # Use the new matplotlib API for colormaps
        try:
            cmap = plt.colormaps[colormap_name]
        except KeyError:
            # Fallback to the old method if the new one doesn't work
            cmap = plt.cm.get_cmap(colormap_name)
    return cmap

def plot_single_tiff(data, title, colormap='viridis', show_histogram=True):
    """Plot a single TIFF file"""
    if data is None:
        return None, None
    
    # Create figure with subplots
    if show_histogram:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{title} - Map', f'{title} - Histogram'),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f'{title} - Map',)
        )
    
    # Normalize data for better visualization
    data_normalized = np.ma.masked_where(data == 0, data)
    
    # Create heatmap with proper colorscale
    heatmap = go.Heatmap(
        z=data_normalized,
        colorscale=colormap,
        showscale=True,
        name=title,
        hoverongaps=False
    )
    
    if show_histogram:
        fig.add_trace(heatmap, row=1, col=1)
        
        # Create histogram
        hist_data = data[data > 0].flatten()  # Remove zeros for histogram
        if len(hist_data) > 0:
            hist, bins = np.histogram(hist_data, bins=50)
            fig.add_trace(
                go.Bar(x=bins[:-1], y=hist, name='Distribution'),
                row=1, col=2
            )
    else:
        fig.add_trace(heatmap)
    
    # Update layout
    fig.update_layout(
        title=f"Visualization: {title}",
        height=600 if show_histogram else 500,
        showlegend=False,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig, data

def plot_overlay_tiffs(tiff_data_dict, colormap='viridis', alpha=0.7):
    """Plot multiple TIFF files as overlay"""
    if not tiff_data_dict:
        return None
    
    fig = go.Figure()
    
    # Define colors for different layers
    colors = ['viridis', 'plasma', 'inferno', 'magma', 'turbo']
    
    for i, (name, data) in enumerate(tiff_data_dict.items()):
        if data is not None:
            # Normalize data
            data_normalized = np.ma.masked_where(data == 0, data)
            
            # Create heatmap for each layer
            heatmap = go.Heatmap(
                z=data_normalized,
                colorscale=colors[i % len(colors)],
                showscale=True,
                name=name,
                opacity=alpha,
                hoverongaps=False
            )
            fig.add_trace(heatmap)
    
    fig.update_layout(
        title="Overlay Visualization of All TIFF Files",
        height=600,
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig

def get_file_info(file_path):
    """Get basic information about a TIFF file"""
    try:
        with rasterio.open(file_path) as src:
            return {
                'filename': os.path.basename(file_path),
                'shape': src.shape,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'bounds': src.bounds,
                'nodata': src.nodata
            }
    except Exception as e:
        return {'error': str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🗺️ TIFF Data Visualizer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Get TIFF files
    tiff_files = get_tiff_files()
    
    if not tiff_files:
        st.error("No TIFF files found in the datasets folder!")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # File selection
    file_options = {os.path.basename(f): f for f in tiff_files}
    selected_file = st.sidebar.selectbox(
        "Select TIFF File:",
        options=list(file_options.keys()),
        index=0
    )
    
    # Visualization mode
    viz_mode = st.sidebar.radio(
        "Visualization Mode:",
        ["Single File", "Overlay All Files"],
        index=0
    )
    
    # Colormap selection
    colormap_options = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'custom_blue']
    selected_colormap = st.sidebar.selectbox(
        "Color Map:",
        options=colormap_options,
        index=0
    )
    
    # Additional options
    show_histogram = st.sidebar.checkbox("Show Histogram", value=True)
    alpha = st.sidebar.slider("Overlay Transparency", 0.1, 1.0, 0.7, 0.1)
    
    # Main content area
    if viz_mode == "Single File":
        st.header(f"📊 {selected_file}")
        
        # Load and display file info
        file_path = file_options[selected_file]
        file_info = get_file_info(file_path)
        
        if 'error' not in file_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dimensions", f"{file_info['shape'][0]} × {file_info['shape'][1]}")
            with col2:
                st.metric("Data Type", file_info['dtype'])
            with col3:
                st.metric("CRS", file_info['crs'] if file_info['crs'] else "Unknown")
        
        # Load and visualize the file
        data, transform, crs, bounds, meta = load_tiff_file(file_path)
        
        if data is not None:
            # Display statistics
            st.subheader("📈 Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Value", f"{np.nanmin(data):.2f}")
            with col2:
                st.metric("Max Value", f"{np.nanmax(data):.2f}")
            with col3:
                st.metric("Mean Value", f"{np.nanmean(data):.2f}")
            with col4:
                st.metric("Std Dev", f"{np.nanstd(data):.2f}")
            
            # Create visualization
            fig, _ = plot_single_tiff(data, selected_file, selected_colormap, show_histogram)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            st.subheader("💾 Download Data")
            if st.button("Download Statistics as CSV"):
                stats_df = {
                    'Statistic': ['Min', 'Max', 'Mean', 'Std Dev', 'Count'],
                    'Value': [
                        np.nanmin(data),
                        np.nanmax(data),
                        np.nanmean(data),
                        np.nanstd(data),
                        np.count_nonzero(data)
                    ]
                }
                import pandas as pd
                df = pd.DataFrame(stats_df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_file}_statistics.csv",
                    mime="text/csv"
                )
    
    else:  # Overlay mode
        st.header("🔍 Overlay Visualization")
        
        # Load all files
        tiff_data = {}
        file_info_dict = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, file_path) in enumerate(file_options.items()):
            status_text.text(f"Loading {name}...")
            data, _, _, _, _ = load_tiff_file(file_path)
            tiff_data[name] = data
            file_info_dict[name] = get_file_info(file_path)
            progress_bar.progress((i + 1) / len(file_options))
        
        status_text.text("Creating overlay visualization...")
        
        # Create overlay plot
        fig = plot_overlay_tiffs(tiff_data, selected_colormap, alpha)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display file information
        st.subheader("📋 File Information")
        for name, info in file_info_dict.items():
            if 'error' not in info:
                with st.expander(f"📄 {name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Dimensions:** {info['shape']}")
                        st.write(f"**Data Type:** {info['dtype']}")
                    with col2:
                        st.write(f"**CRS:** {info['crs'] if info['crs'] else 'Unknown'}")
                        st.write(f"**No Data Value:** {info['nodata']}")
        
        status_text.text("✅ Overlay visualization complete!")
        progress_bar.empty()
        status_text.empty()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🗺️ TIFF Data Visualizer | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
