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
- Export optimized routes to dual-format output files (JSONL + Parquet)
- Store results in structured formats for different use cases
- Prepare data for downstream systems and analytics dashboards

## Project Structure

```
route-optimization/
├── main.py                           # Main entry point and CLI interface
├── config/
│   └── model-params.yaml             # Optimization parameters
├── data/
│   └── locations.jsonl               # Input location data
├── output/                           # Generated results (dual format: JSONL + Parquet)
│   ├── itinerary.jsonl               # Individual position records (JSONL)
│   ├── itinerary.parquet             # Individual position records (Parquet)
│   ├── daily-summary.jsonl           # Daily aggregated metrics (JSONL)
│   ├── daily-summary.parquet         # Daily aggregated metrics (Parquet)
│   ├── zone-summary.jsonl            # Zone-level analytics (JSONL)
│   ├── zone-summary.parquet          # Zone-level analytics (Parquet)
│   ├── aggregate-summary.jsonl       # System-wide statistics (JSONL)
│   └── aggregate-summary.parquet     # System-wide statistics (Parquet)
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
uv run python main.py --clusterer dbscan              # DBSCAN clustering
uv run python main.py --clusterer balanced            # Balanced clustering

# Choose balancing approach for workload equalization
uv run python main.py --balancer greedy               # Default: Enhanced greedy transfer
uv run python main.py --balancer simulated_annealing  # Simulated annealing
uv run python main.py --balancer min_max              # Min-max optimization
uv run python main.py --balancer local_search         # Local search optimization
uv run python main.py --balancer network_flow         # Network flow optimization

# Multi-algorithm comparison
uv run python main.py --clusterer mds_kmeans dbscan --balancer greedy min_max
uv run python main.py --full-grid                     # Run all algorithm combinations
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

### API Endpoints

The OSRM server provides a comprehensive HTTP API for routing and navigation services:

#### Base URLs
```bash
# Kubernetes (external access)
http://192.168.50.2:32050

# Docker Compose (local development)
http://localhost:5000
```

#### Available services

**1. Route service - `/route/v1/driving/{coordinates}`**

The route service calculates the fastest route between coordinates with detailed navigation information.

```bash
# Basic route between two points (SF to Oakland)
curl "http://192.168.50.2:32050/route/v1/driving/-122.4194,37.7749;-122.2711,37.8044"
# Returns: Route geometry, duration (1174.2s ≈ 19.6min), distance (18.1km)

# Route with turn-by-turn navigation steps
curl "http://192.168.50.2:32050/route/v1/driving/-122.4194,37.7749;-122.2711,37.8044?steps=true"
# Returns: Detailed turn-by-turn instructions with intersections and maneuvers

# Route with alternative paths
curl "http://192.168.50.2:32050/route/v1/driving/-122.4194,37.7749;-122.2711,37.8044?alternatives=true"
# Returns: Multiple route options with different paths

# Route without geometry overview (faster response)
curl "http://192.168.50.2:32050/route/v1/driving/-122.4194,37.7749;-122.2711,37.8044?overview=false"
# Returns: Route info without detailed geometry polyline

# Multi-waypoint route (SF → Oakland → San Jose)
curl "http://192.168.50.2:32050/route/v1/driving/-122.4194,37.7749;-122.2711,37.8044;-121.8863,37.3382"
# Returns: Route through multiple waypoints with leg-by-leg breakdown
```

**2. Nearest service - `/nearest/v1/driving/{coordinates}`**

Finds the nearest road network point to given coordinates.

```bash
# Find nearest road point to SF coordinates
curl "http://192.168.50.2:32050/nearest/v1/driving/-122.4194,37.7749"
# Returns: Nearest routable point with node IDs and distance offset

# Find 3 nearest road points
curl "http://192.168.50.2:32050/nearest/v1/driving/-122.4194,37.7749?number=3"
# Returns: Array of 3 closest routable points
```

**3. Table service - `/table/v1/driving/{coordinates}` (origin-destination matrix)**

Computes distance and duration matrices between multiple points. This service generates **origin-destination matrices** that show travel times and distances from every origin to every destination - perfect for logistics, route optimization, fleet management, and delivery planning.

```bash
# Distance/duration matrix for 3 points (SF, Oakland, San Jose)
curl "http://192.168.50.2:32050/table/v1/driving/-122.4194,37.7749;-122.2711,37.8044;-121.8863,37.3382"
# Returns: Full 3x3 origin-destination matrix of distances and durations between all point pairs

# Matrix with specific source (from SF to all destinations)
curl "http://192.168.50.2:32050/table/v1/driving/-122.4194,37.7749;-122.2711,37.8044;-121.8863,37.3382?sources=0"
# Returns: 1x3 matrix from SF to all other points

# Matrix with distance and duration annotations
curl "http://192.168.50.2:32050/table/v1/driving/-122.4194,37.7749;-122.2711,37.8044?annotations=distance,duration"
# Returns: Separate distance and duration arrays for analysis
```

**4. Match service - `/match/v1/driving/{coordinates}` (GPS trace matching)**

Matches GPS traces to road network - essential for cleaning noisy GPS data.

```bash
# GPS trace matching with noisy coordinates
curl "http://192.168.50.2:32050/match/v1/driving/-122.4194,37.7749;-122.4190,37.7750;-122.4185,37.7751;-122.2711,37.8044"
# Returns: Cleaned route snapped to road network with confidence scores

# Match with full overview and turn-by-turn steps
curl "http://192.168.50.2:32050/match/v1/driving/-122.4194,37.7749;-122.4190,37.7750;-122.4185,37.7751;-122.2711,37.8044?overview=full&steps=true"
# Returns: Full route geometry and detailed navigation instructions
```

**5. Trip service - `/trip/v1/driving/{coordinates}` (traveling salesman optimization)**

Solves traveling salesman problem to find optimal visit order for multiple locations.

```bash
# Trip optimization for 4 cities (finds optimal order)
curl "http://192.168.50.2:32050/trip/v1/driving/-122.4194,37.7749;-122.2711,37.8044;-121.8863,37.3382;-122.2585,37.8716"
# Returns: Optimized route visiting all points with minimal total distance

# Trip with fixed start and end points
curl "http://192.168.50.2:32050/trip/v1/driving/-122.4194,37.7749;-122.2711,37.8044;-121.8863,37.3382?source=first&destination=last"
# Returns: Optimized route starting at first point, ending at last point
```

**6. Tile service - `/tile/v1/driving/tile(x,y,z).mvt` (debug vector tiles)**

Generates Mapbox Vector Tiles for debugging and visualizing the routing graph.

```bash
# Vector tile for San Francisco area (zoom level 13)
curl "http://192.168.50.2:32050/tile/v1/driving/tile(1310,3166,13).mvt"
# Returns: Binary Mapbox Vector Tile (.mvt) with routing graph data

# Notes:
# - Requires zoom level 12 or higher (OSRM limitation)
# - URL format: tile(x,y,z).mvt where x,y,z follow standard web map tile coordinates
# - Returns binary protobuf data (application/x-protobuf)
# - Primarily for debugging routing graphs, not optimized for general map rendering
# - Contains two layers: 'speeds' and 'turns' with road geometries and metadata
```

#### Common parameters

- `overview`: Geometry overview (`full`, `simplified`, `false`)
- `steps`: Include turn-by-turn navigation (`true`, `false`)
- `alternatives`: Return alternative routes (`true`, `false`)
- `annotations`: Additional metadata (`duration`, `distance`, `speed`)
- `geometries`: Response geometry format (`polyline`, `polyline6`, `geojson`)

#### Response format

All endpoints return JSON responses with this structure:
```json
{
  "code": "Ok",
  "routes": [...],
  "waypoints": [...]
}
```

### Example coordinates (California)

The test suite and examples use these California coordinates:
- **San Francisco**: `-122.4194,37.7749`
- **Oakland**: `-122.2711,37.8044`
- **San Jose**: `-121.8863,37.3382`
- **Berkeley**: `-122.2585,37.8716`

### API documentation

For detailed API documentation and additional parameters:
- **Official OSRM API docs**: https://project-osrm.org/docs/v5.24.0/api/
- **GitHub documentation**: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md

## Performance

Multi-threaded processing characteristics:
- **Total Runtime**: ~6.36 seconds for 24 zones
- **Threading**: 12 worker threads (CPU count / 2)
- **Performance Gain**: 3.58x faster than sequential
- **Optimization Quality**: 67% improved workload balance

### Multi-Algorithm Orchestration

The system supports comprehensive algorithm comparison through multi-algorithm orchestration:

- **Grid Search**: Test all combinations of clustering and balancing algorithms
- **Sequential Execution**: Run experiments sequentially for reliable results
- **Smart Merging**: Combine new results with existing data, keeping newest records
- **Performance**: 25 algorithm combinations (5 clusterers × 5 balancers) complete in ~2-3 minutes

Available algorithms:
- **Clusterers**: `mds_kmeans`, `hierarchical`, `spectral`, `dbscan`, `balanced` (5 total)
- **Balancers**: `greedy`, `simulated_annealing`, `min_max`, `local_search`, `network_flow` (5 total)

## Data Export Formats

The system exports all output data in **dual formats** to optimize for different use cases:

### JSONL Format
- **Use case**: Human-readable debugging, data inspection, and external tool integration
- **Characteristics**: Line-delimited JSON for easy streaming and manual inspection
- **Files**: `itinerary.jsonl`, `daily-summary.jsonl`, `zone-summary.jsonl`, `aggregate-summary.jsonl`

### Parquet Format
- **Use case**: Fast analytics queries, Streamlit dashboards, and data science workflows
- **Characteristics**: Columnar binary format optimized for analytical workloads
- **Performance**: ~10x faster reads for large datasets compared to JSONL
- **Files**: `itinerary.parquet`, `daily-summary.parquet`, `zone-summary.parquet`, `aggregate-summary.parquet`

### Smart Merging & Processing Optimization
The system uses Parquet as the primary format for all data operations:
- **Primary format**: All upsert operations performed on Parquet files using Polars for maximum speed
- **Legacy migration**: Automatically migrates from JSONL-only datasets on first run
- **Processing order**: Read Parquet → Perform upsert → Write Parquet → Write JSONL backup
- **Conflict resolution**: Keeps newest records based on `created_on` timestamp
- **Uniqueness**: Maintains data integrity per uniqueness constraints documented below
- **Performance**: ~10x faster processing compared to JSONL-based operations

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run python -m black .

# Type checking  
uv run python -m mypy src/
```

## data models

### input 

#### locations.jsonl

Each line in the JSONL file represents a single location with the following fields:

| Field       | Type    | Description                                              | Example                                     |
|-------------|---------|----------------------------------------------------------|---------------------------------------------|
| `pos_id`    | integer | Unique identifier for the location                       | `31`                                        |
| `name`      | string  | Location name (typically "Subway - [City]")              | `"Subway - Buttonwillow"`                   |
| `address`   | string  | Full street address                                      | `"20673 Tracy Ave, Buttonwillow, CA"`       |
| `latitude`  | float   | Geographic latitude coordinate                           | `35.40011`                                  |
| `longitude` | float   | Geographic longitude coordinate                          | `-119.3978`                                 |
| `zone_id`   | string  | Zone identifier for grouping locations (nullable)        | `"zone_012"`                                |
| `class`     | string  | Location classification ("primary" or "secondary")       | `"primary"`                                 |

- unique to (will hold only 1 unqiue combination of):
  - `zone_id`
  - `pos_id`
  
### output

#### itinerary.jsonl

Each line represents an individual position record with action details:

| Field            | Type         | Description                                           | Example                                      |
|------------------|--------------|-------------------------------------------------------|----------------------------------------------|
| `zone_id`        | string       | Zone identifier                                       | `"zone_017"`                                 |
| `day`            | integer      | Day of the week (1-7)                                 | `2`                                          |
| `pos_id`         | string       | String value of individual pos id (null for centroid) | `"302468"`                                   |
| `pos_name`       | string       | name value associated with pos_id                     | `"Subway - Kettleman City"`                  |
| `pos_class`      | string       | Classification of each location in route              | `"secondary"`                                |
| `route`          | array[float] | Each lon, lat point along the path                    | `[[-120.97, 40.62]]`                         |
| `action`         | string       | The action taken at each point along the route        | `"driving"`                                  |
| `schedule`       | float        | Time in float minutes at the start of each action     | `4.222`                                      |
| `clusterer`      | string       | Clustering algorithm used for creation                | `"mds_kmeans"`                               |
| `router`         | string       | Routing algorithm used for individual days            | `"exhaustive"`                               |
| `balancer`       | string       | Method used for cluster balancing                     | `"greedy"`                                   |
| `created_on`     | datetime     | Timestamp at which the elements were written to file  | `"2025-09-09 09:55:27"`                      |

**Action Types:**
- `driving`: Vehicle is traveling between locations
- `arriving`: Vehicle is arriving at a location
- `departing`: Vehicle is departing from a location

**Scheduling Rules:**
- Each day starts at schedule = 0.0 with action = "driving" from centroid (pos_id = null)
- Arrival and departure times are recorded for each location
- Each day ends with action = "departing" from the final location

**Unique Constraints:**
- `zone_id`, `pos_id`, `day`, `clusterer`, `balancer`

#### daily-summary.jsonl

Daily aggregation of position records by zone and day:

| Field                    | Type     | Description                                           | Example               |
|--------------------------|----------|-------------------------------------------------------|-----------------------|
| `zone_id`                | string   | Zone identifier                                       | `"zone_017"`          |
| `day`                    | integer  | Day of the week (1-7)                                 | `2`                   |
| `primary_locations`      | integer  | Total number of primary locations visited on that day | `1`                   |
| `secondary_locations`    | integer  | Total number of secondary locations visited           | `5`                   |
| `duration`               | float    | Total duration of all activities on a given day (min) | `8.647815`            |
| `utilization_percentage` | float    | Percentage of time compared to `hours_per_day` param  | `94.876`              |
| `total_pos_time`         | float    | Total minutes spent at all POS locations             | `846.0000`            |
| `total_drive_time`       | float    | Total minutes spent driving on a given day           | `46.486118`           |
| `clusterer`              | string   | Clustering algorithm used for creation                | `"mds_kmeans"`        |
| `router`                 | string   | Routing algorithm used for individual days            | `"exhaustive"`        |
| `balancer`               | string   | Method used for cluster balancing                     | `"greedy"`            |
| `created_on`             | datetime | Timestamp at which the elements were written to file | `"2025-09-09 09:55:27"` |

**Unique Constraints:**
- `zone_id`, `day`, `clusterer`, `balancer`

#### zone-summary.jsonl

Zone-level performance metrics and statistics:

| Field                  | Type     | Description                                          | Example               |
|------------------------|----------|------------------------------------------------------|-----------------------|
| `zone_id`              | string   | Zone identifier                                      | `"zone_000"`          |
| `primary_count`        | integer  | Number of primary locations in zone                  | `2`                   |
| `secondary_count`      | integer  | Number of secondary locations in zone                | `10`                  |
| `weekly_duration`      | float    | Total weekly route duration (hours)                  | `6.47725`             |
| `utilization`          | float    | Average capacity utilization percentage              | `16.19`               |
| `overutilized_days`    | integer  | Count of days exceeding capacity                     | `0`                   |
| `underutilized_days`   | integer  | Count of days below optimal capacity                 | `3`                   |
| `total_pos_time`       | float    | Total time spent at locations (hours)                | `12.0`                |
| `total_drive_time`     | float    | Total driving time (hours)                           | `0.0`                 |
| `duration_std`         | float    | Standard deviation of all daily durations            | `0.78`                |
| `clusterer`            | string   | Clustering algorithm used for creation               | `"mds_kmeans"`        |
| `router`               | string   | Routing algorithm used for individual days           | `"exhaustive"`        |
| `balancer`             | string   | Method used for cluster balancing                    | `"greedy"`            |
| `created_on`           | datetime | Timestamp at which the elements were written to file | `"2025-09-09 09:55:27"` |

**Unique Constraints:**
- `zone_id`, `clusterer`, `balancer`

#### aggregate-summary.jsonl

System-wide summary statistics (single record):

| Field                                          | Type     | Description                                           | Example               |
|------------------------------------------------|----------|-------------------------------------------------------|-----------------------|
| `average_weekly_duration`                      | float    | Average weekly duration across all zones              | `0.0`                 |
| `average_utilization`                          | float    | Average utilization across all zones                  | `0.0`                 |
| `average_overutilized_days`                    | float    | Average overutilized days per zone                    | `0.0`                 |
| `average_underutilized_days`                   | float    | Average underutilized days per zone                   | `3.71`                |
| `average_daily_pos_time`                       | float    | Average daily time at locations (hours)               | `14.64`               |
| `average_daily_drive_time`                     | float    | Average daily driving time (hours)                    | `0.0`                 |
| `average_duration_standard_deviation`          | float    | Average std dev of all daily durations                | `0.0`                 |
| `clusterer`                                    | string   | Clustering algorithm used for creation                | `"mds_kmeans"`        |
| `router`                                       | string   | Routing algorithm used for individual days            | `"exhaustive"`        |
| `balancer`                                     | string   | Method used for cluster balancing                     | `"greedy"`            |
| `created_on`                                   | datetime | Timestamp at which the elements were written to file | `"2025-09-09 09:55:27"` |

**Unique Constraints:**
- `clusterer`, `balancer`