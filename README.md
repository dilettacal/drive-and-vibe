# Drive-and-Vibe 🚗☕

**Geolife Dataset**

A hands-on, lightweight data science project that analyzes GPS trajectories to model driving behavior patterns. Think autonomous driving research, but scaled down to something you can run on a laptop with just GPS data and some Python libraries.

## 🎯 What This Project Does

This project analyzes real GPS trajectories from the Geolife dataset to:

- **Detect driving patterns** — Identify different driving styles (aggressive, normal, cautious)
- **Classify behaviors** — Use clustering to group similar trips
- **Find interesting events** — Detect hard braking, sudden acceleration, stops, and sharp turns
- **Visualize insights** — Create interactive maps and timeline plots

No neural networks, no cameras, no LiDAR — just math, pandas, and vibes. ☕

## 🧱 Project Structure

```
drive-and-vibe/
│
├── data/
│   ├── geolife/                     # Input .plt files (downloaded separately)
│   └── processed/                   # Cleaned data and features
│
├── notebooks/
│   ├── 01_ingest_and_clean.ipynb     # Load and clean GPS data
│   ├── 02_feature_engineering.ipynb  # Extract trip features
│   ├── 03_behavior_clustering.ipynb  # Cluster driving behaviors
│   ├── 04_event_detection.ipynb      # Detect driving events
│   └── 05_visualization.ipynb        # Create visualizations
│
├── src/
│   ├── __init__.py
│   ├── geo_utils.py                  # GPS data utilities
│   ├── feature_engineering.py        # Feature extraction
│   ├── clustering.py                 # Behavior clustering
│   ├── events.py                     # Event detection
│   └── visualization.py              # Plotting functions
│
├── scripts/
│   └── download_geolife.py           # Download dataset
│
├── tests/
│   └── test_geo_utils.py             # Unit tests
│
├── outputs/                          # Generated visualizations
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd drive-and-vibe
   ```

2. **Install dependencies**:
   ```bash
   make install
   # or manually: uv sync
   ```
   
   This will create a virtual environment managed by UV and install all dependencies.

3. **Set up pre-commit hooks** (optional):
   ```bash
   make setup
   ```

4. **Download the Geolife dataset**:
   ```bash
   make data
   # or: python scripts/download_geolife.py
   ```

### Usage

Run the analysis pipeline:

1. **Open Jupyter**:
   ```bash
   make jupyter
   ```

2. **Run notebooks in sequence**:
   - `01_ingest_and_clean.ipynb` — Load and clean GPS data
   - `02_feature_engineering.ipynb` — Extract features
   - `03_behavior_clustering.ipynb` — Cluster behaviors
   - `04_event_detection.ipynb` — Detect events
   - `05_visualization.ipynb` — Create visualizations

Or run all at once:
```bash
make run-all
```

## 📊 What You'll Get

After running the notebooks, you'll have:

- **Cleaned trajectory data** (`data/processed/trajectories_cleaned.parquet`)
- **Trip features** (`data/processed/trip_features.csv`)
- **Behavior clusters** (`data/processed/trip_clusters.csv`)
- **Event markers** (`data/processed/trajectories_with_events.parquet`)
- **Interactive maps** (`outputs/trajectory_map.html`)
- **Visualization plots** (`outputs/behavior_clusters.png`, etc.)

## 🧰 Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies with uv |
| `make install-dev` | Install dev dependencies |
| `make setup` | Set up pre-commit hooks |
| `make lint` | Run linting |
| `make format` | Format code |
| `make test` | Run tests |
| `make data` | Download Geolife dataset |
| `make jupyter` | Launch Jupyter |
| `make run-all` | Run all notebooks |
| `make clean` | Clean generated files |

## 🧠 Core Concepts

### Behavior Clustering

Trips are classified into three behavior types:

- **🚙 Aggressive** — Fast driving, high acceleration variation, quick speeds
- **🚗 Normal** — Balanced driving characteristics
- **🚶 Cautious** — Slower speeds, more stops, smoother acceleration

### Event Detection

The following events are detected:

- **Hard braking** — Deceleration < -3 m/s²
- **Sudden acceleration** — Acceleration > +3 m/s²
- **Stops** — Speed < 1 km/h
- **Sharp turns** — Heading change > 30°

### Feature Engineering

For each trip, we compute:

- Speed statistics (mean, std, max)
- Acceleration statistics (mean, std, max, min)
- Percent of time stopped
- Trip duration and distance
- Jerk (smoothness indicator)
- Speed consistency

## 🔬 Technical Details

### Libraries Used

- **pandas** — Data manipulation
- **numpy** — Numerical computations
- **geopy** — GPS distance calculations
- **scikit-learn** — Clustering (KMeans, DBSCAN)
- **folium** — Interactive maps
- **matplotlib/seaborn** — Static plots

### Data Processing Pipeline

1. **Loading** — Parse .plt files from Geolife dataset
2. **Speed/Acceleration** — Compute from GPS coordinates using Haversine distance
3. **Filtering** — Remove unrealistic points (teleporting, impossible speeds)
4. **Feature Extraction** — Aggregate statistics per trip
5. **Clustering** — Standardize features and cluster
6. **Event Detection** — Identify driving events
7. **Visualization** — Generate maps and plots

## 📝 Development

### Running Tests

```bash
make test
```

### Code Quality

```bash
make lint    # Check for issues
make format  # Auto-fix formatting
```

### Adding New Features

The project is organized into modular components:

- `src/geo_utils.py` — GPS data handling
- `src/feature_engineering.py` — Feature extraction
- `src/clustering.py` — Behavior clustering
- `src/events.py` — Event detection
- `src/visualization.py` — Plotting

Each module is independent and can be extended or replaced.

## 📚 Data Source

This project uses the [Geolife GPS Trajectory Dataset](https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset) from Kaggle (originally from Microsoft Research).

> Zheng Y, Xie X, Ma W. Geolife: a collaborative social networking service among user, location and trajectory. IEEE Data(base) Engineering Bulletin, 2010.

### Dataset Information

- **18,670 GPS trajectory files** (.plt format)
- **182 users** tracked over 5 years (2007-2012)
- **1.3 million kilometers** of total trajectory data
- **50,000+ hours** of GPS tracking

The dataset will be automatically downloaded when you run `make data`.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make lint` and `make test`
5. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft Research for the Geolife dataset
- The open source community for the amazing tools

---

**Made with ☕ and good vibes** 🚗

