# UE Localization Web Interface

Beautiful, minimal-code visualization interface built with Streamlit.

## Features

- 🗺️ **Interactive Map Viewer** - Visualize predictions, radio maps, and building overlays
- 📊 **Performance Dashboard** - Comprehensive metrics with CDF plots and comparison tables  
- 🔍 **Live Inference Mode** - Upload measurements and get real-time predictions
- 📈 **Analysis Tools** - Feature importance, error analysis, ablation studies
- 🎨 **Beautiful UI** - Modern, clean interface with minimal code

## Quick Start

### Installation

```bash
# Navigate to web directory
cd web

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# From the web directory
streamlit run streamlit_app.py

# Or from project root
streamlit run web/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### 1. Overview Page
- View system architecture
- Check model status
- See quick statistics

### 2. Metrics Dashboard
- Error CDF plots
- Percentile tables
- Model comparison
- Success rates

### 3. Live Inference
- Generate demo measurements
- Upload JSON files
- Run real-time inference
- Visualize predictions

### 4. Analysis
- Feature importance (SHAP values)
- Error analysis by scenario
- Ablation study results

## Input Format

Upload measurements as JSON:

```json
{
  "scene_id": "demo_city_tile_01",
  "measurements": [
    {
      "timestamp": 0.0,
      "cell_id": 101,
      "beam_id": 3,
      "rsrp": -82.3,
      "rsrq": -11.5,
      "sinr": 14.2,
      "cqi": 10,
      "ta": 98,
      "aoa_azimuth": 42.5
    }
  ]
}
```

## Configuration

### Model Path
Edit `streamlit_app.py` line 58 to point to your trained model:
```python
model_path = Path("checkpoints/best_model.pt")
```

### Data Paths
The app looks for:
- `data/test_metrics.json` - Performance metrics
- `data/scenes_metadata.json` - Scene information
- `data/radio_maps.zarr` - Precomputed radio maps (optional)
- `data/osm_maps.zarr` - OSM building maps (optional)

## Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment

#### Option 1: Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

#### Option 2: Docker
```bash
# Build image
docker build -t ue-localization-web .

# Run container
docker run -p 8501:8501 ue-localization-web
```

#### Option 3: Custom Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run with custom port and config
streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0
```

## Customization

### Theme
Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Page Config
Modify in `streamlit_app.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🎯",
    layout="wide"
)
```

## Performance

- **Cold start**: ~2-3 seconds
- **Inference**: 40-50ms on GPU, 100-200ms on CPU
- **Page load**: <1 second
- **Memory**: ~500MB with model loaded

## Troubleshooting

### Model not found
```
Error: Model not found at checkpoints/best_model.pt
```
Solution: Train a model or use demo mode (automatic)

### Import errors
```
ModuleNotFoundError: No module named 'streamlit'
```
Solution: `pip install -r requirements.txt`

### Port already in use
```
Error: Address already in use
```
Solution: `streamlit run streamlit_app.py --server.port 8502`

## Advanced Features

### Add Custom Pages
```python
# In streamlit_app.py
def show_custom_page():
    st.header("Custom Analysis")
    # Your code here

# Add to navigation
page = st.radio("Navigation", [..., "Custom Page"])
if page == "Custom Page":
    show_custom_page()
```

### Add Map Layers
Use folium for interactive maps:
```python
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[40.0, -105.0], zoom_start=13)
folium.Marker([40.0, -105.0], popup="Prediction").add_to(m)
st_folium(m, width=700, height=500)
```

### Add Real-time Updates
Use `st.empty()` for live updates:
```python
placeholder = st.empty()
for i in range(100):
    placeholder.metric("Live Value", i)
    time.sleep(0.1)
```

## License

MIT License - see LICENSE file

## Support

For issues and questions:
- GitHub Issues: [repository link]
- Documentation: See main project README
