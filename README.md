# Route Optimization System

A comprehensive Python-based route optimization system for solving vehicle routing problems (VRP) with real-world driving data. The system integrates with a local OSRM (Open Source Routing Machine) server for accurate distance matrices and turn-by-turn routing.

## System Overview

The system follows a structured pipeline approach:

### 1. Extraction
- Load location data from JSONL files
- Validate coordinates and location metadata
- Filter locations by zone assignments

### 2. Preprocessing
- Clean and normalize location data
- Calculate geographic centroids
- Generate distance matrices via OSRM

### 3. Optimization

#### 3.1 Primary Day Assignment
- Assign primary (high-value) locations to dedicated days
- Allocate 8 hours per primary location
- Distribute workload across available days

#### 3.2 Secondary Day Clustering
- K-means clustering for secondary locations
- Noise detection for isolated locations (>150km)
- Spatial grouping based on geographic proximity

#### 3.3 Route Optimization
- Adaptive TSP algorithms (exhaustive for ≤5 locations, greedy+2-opt for larger)
- Minimize total drive time per day
- Generate optimal visit sequences

#### 3.4 Cluster Balancing
- Organic duration rebalancing across clusters
- Workload equalization (60-minute threshold)
- Iterative location reassignment

#### 3.5 Detailed Routing
- Fetch turn-by-turn directions from OSRM
- Generate route geometry polylines
- Calculate accurate drive times and distances

### 4. Reporting
- Generate comprehensive analytics
- Export JSON reports with itineraries
- Create interactive HTML visualizations
- Calculate utilization metrics

### 5. Loading
- Export optimized routes to output files
- Store results in structured formats
- Prepare data for downstream systems

## Project Structure

```
route-optimization/
├── main.py                           # Main entry point and CLI interface
├── config/
│   └── model-params.yaml             # Optimization parameters
├── data/
│   └── locations.jsonl               # Input location data
├── output/                           # Generated results
│   ├── aggregate-report.jsonl        # Cross-zone analytics
│   ├── aggregate-summary.jsonl       # Summary statistics  
│   └── itinerary.jsonl               # Daily route itineraries
├── src/
│   ├── core/                         # Core optimization algorithms
│   │   ├── optimize.py               # Main optimization engine
│   │   └── report.py                 # Reporting and analytics
│   └── utils/                        # Utility modules
│       ├── clustering_utils.py       # K-means clustering
│       ├── geo_utils.py              # Geographic utilities
│       ├── osrm_utils.py             # OSRM API integration
│       └── io_utils.py               # Data I/O utilities
├── streamlit/                        # Web application
│   └── app.py                        # Interactive dashboard
└── tests/                            # Test suite
```

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Usage

### Single Zone Optimization
```bash
# Optimize a specific zone
uv run python main.py zone_000

# View results
open output/visualizations/route_map_zone_000.html
```

### Multi-Zone Optimization  
```bash
# Optimize all zones concurrently
uv run python main.py

# Limit concurrent workers and zones
uv run python main.py --workers 4 --zones 10
```

### Algorithm Selection
```bash
# Choose clustering algorithm for secondary location grouping
uv run python main.py --clusterer mds_kmeans           # Default: MDS + K-means
uv run python main.py --clusterer hierarchical         # Hierarchical clustering
uv run python main.py --clusterer spectral            # Spectral clustering

# Choose balancing approach for workload equalization
uv run python main.py --balancer greedy               # Default: Enhanced greedy transfer
uv run python main.py --balancer simulated_annealing  # Simulated annealing
uv run python main.py --balancer min_max              # Min-max optimization
```

### Clustering Operations
```bash
# Re-cluster locations with custom parameters
uv run python main.py --cluster --min 5 --max 20

# Fix coordinates via geocoding
uv run python main.py --geocode
```

## Configuration

The system uses `config/model-params.yaml`:
- `primary_hours_per_location`: Hours for primary stores (default: 8.0)
- `secondary_hours_per_location`: Hours for secondary stores (default: 1.0) 
- `working_days_per_week`: Available days (default: 5)
- `max_locations_per_day`: Daily location limit (default: 7)

## OSRM Integration

Local OSRM server configuration:
- **Base URL**: `http://192.168.50.2:32050`
- **Profile**: Driving profile (California road network)
- **APIs**: Table (distance matrix) and Route (geometry)

## Performance

Multi-threaded processing characteristics:
- **Total Runtime**: ~6.36 seconds for 24 zones
- **Threading**: 12 worker threads (CPU count / 2)
- **Performance Gain**: 3.58x faster than sequential
- **Optimization Quality**: 67% improved workload balance

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run python -m black .

# Type checking  
uv run python -m mypy src/
```

## Data Formats

### Input (JSONL)
```json
{
  "pos_id": 1,
  "name": "Subway - Granada Hills",
  "address": "11878 Balboa Blvd, Granada Hills, CA",
  "latitude": 34.2776949,
  "longitude": -118.502159,
  "zone_id": "zone_000",
  "class": "primary"
}
```

### Output (JSON)
- Detailed itineraries with route sequences
- Metrics summaries with drive times
- Route geometries with polylines
- Turn-by-turn instructions