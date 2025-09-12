# Route Optimization Methodology

This document outlines the data science methodologies, algorithms, and mathematical assumptions underlying the route optimization system.

## Problem Formulation

The system addresses route optimization as a **multi-stage pipeline** with hierarchical decomposition, enabling tractable solutions to the Vehicle Routing Problem (VRP) variant with time windows and capacity constraints.

### Core Assumptions

#### Location Classification
- **Primary Locations**: High-value locations requiring extended time investment (8 hours)
- **Secondary Locations**: Standard locations with fixed service times (1 hour)

#### Time Allocation Model
- Primary locations consume full days based on weekly hour allocation
- Secondary locations have fixed service duration
- Work capacity constraint: maximum 7 locations per day
- Weekly schedule: 5 working days

## Pipeline Stages

### 1. Extraction

**Data Ingestion Process**:
- Load locations from JSONL files
- Parse coordinate data (latitude/longitude)
- Extract location metadata (name, address, zone_id, class)
- Filter invalid or incomplete records

**Input Schema Validation**:
```json
{
  "pos_id": integer,
  "name": string,
  "address": string,
  "latitude": float (-90 to 90),
  "longitude": float (-180 to 180),
  "zone_id": string (nullable),
  "class": "primary" | "secondary"
}
```

### 2. Preprocessing

**Data Normalization**:
- Validate coordinate bounds
- Handle null zone assignments
- Group locations by zone_id
- Calculate zone centroids

**Distance Matrix Generation**:
- Query OSRM Table API for all location pairs
- Include zone centroid as potential starting point
- Build symmetric distance matrix for TSP

**Mathematical Foundation**:
```python
centroid_lat = mean(location_latitudes)
centroid_lon = mean(location_longitudes)
```

### 3. Optimization

#### 3.1 Primary Day Assignment

**Algorithm**: Greedy Time-Based Assignment

**Mathematical Formulation**:
```
Hours per primary = total_primary_hours / num_primary_locations
For each primary location:
    Distribute hours across available days
    Fill days sequentially until all hours allocated
```

**Time Allocation**:
- `h_p = H_total / |P|` where `H_total` = 24 weekly hours, `|P|` = primary locations
- Day utilization: Fill to capacity before moving to next day

#### 3.2 Secondary Day Clustering

**Algorithm**: K-means Clustering with Noise Detection

**Distance Metric**: Haversine distance (great-circle)
```python
haversine_distance = R * 2 * arcsin(sqrt(
    sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
))
```
where R = 6371 km (Earth's radius)

**Clustering Process**:
1. Initialize K clusters using K-means++
2. Assign locations to nearest centroid
3. Update centroids
4. Iterate until convergence

**Noise Detection**:
- Isolation threshold: 150km radius
- Minimum neighbors: 2 locations
- Excluded points: Assigned zone_id: null

#### 3.3 Route Optimization

**Adaptive Algorithm Selection**:
- **Small problems (≤5 locations)**: Exhaustive search
- **Large problems (>5 locations)**: Greedy + 2-opt

**Exhaustive Search**:
```python
for permutation in itertools.permutations(locations):
    route = [start] + list(permutation)
    evaluate route_cost
    keep best_route
```
- Complexity: O(n!)
- Guarantee: Global optimum

**Greedy + 2-opt**:

*Phase 1: Greedy Construction*
```
current = start_location
while unvisited:
    next = argmin(drive_time(current, loc)) for loc in unvisited
    add next to route
    current = next
```

*Phase 2: 2-opt Improvement*
```
for i in range(1, n-2):
    for j in range(i+1, n):
        if swap_improves_cost(i, j):
            reverse_segment(route, i, j)
```
- Complexity: O(n²) per iteration
- Result: Local optimum

#### 3.4 Cluster Balancing

**Algorithm**: Organic Duration Rebalancing

**Objective**: Minimize workload variance across clusters

**Process**:
1. Calculate total duration per cluster
   ```
   duration = Σ(service_times) + Σ(drive_times)
   ```
2. Identify imbalances (60-minute threshold)
3. Move boundary locations between clusters
4. Re-evaluate balance
5. Iterate until convergence (max 5 iterations)

**Rebalancing Criteria**:
- Duration threshold: 60 minutes
- Location selection: Closest to receiving cluster centroid
- Convergence: Std dev of secondary locations < 2.0

#### 3.5 Detailed Routing

**OSRM Route API Integration**:
- Fetch turn-by-turn directions
- Extract route geometry (polyline)
- Calculate segment distances/durations

**Data Structure**:
```json
{
  "geometry": "encoded_polyline",
  "distance": meters,
  "duration": seconds,
  "legs": [
    {
      "steps": [
        {
          "instruction": "Turn left onto Main St",
          "distance": meters,
          "duration": seconds
        }
      ]
    }
  ]
}
```

### 4. Reporting

**Metrics Generation**:
- Total locations visited per zone
- Total drive time and distance
- Daily utilization rates
- Workload balance statistics

**Output Formats**:
1. **Aggregate Report** (aggregate-report.jsonl):
   - Cross-zone analytics
   - Performance comparisons
   - Optimization quality metrics

2. **Summary Statistics** (aggregate-summary.jsonl):
   - High-level KPIs
   - Zone-level summaries
   - System-wide metrics

3. **Detailed Itineraries** (itinerary.jsonl):
   - Day-by-day routes
   - Location visit sequences
   - Timing schedules

**Visualization**:
- Interactive HTML maps with route overlays
- Daily summary tables
- Performance dashboards

### 5. Loading

**Data Export Process**:
- Write optimized routes to JSON files
- Generate HTML visualizations
- Create CSV exports for external systems

**Output Directory Structure**:
```
output/
├── aggregate-report.jsonl
├── aggregate-summary.jsonl
├── itinerary.jsonl
└── visualizations/
    └── route_map_zone_*.html
```

## Mathematical Constraints

### Model Parameters
```yaml
days_per_week: 5                 # Available working days
hours_per_day: 8                 # Daily capacity
primary_hours_per_week: 24      # Total weekly primary hours
hours_per_non_primary: 1        # Service time per secondary
locations_per_day_max: 7        # Maximum locations per day
```

### Constraint Formulation
1. **Capacity**: `Σ(locations_in_day) ≤ 7`
2. **Time**: `Σ(service_times) + Σ(drive_times) ≤ 8 hours`
3. **Primary Hours**: `Σ(primary_hours) = 24 hours/week`
4. **Assignment**: Each location assigned exactly once

## Algorithmic Complexity

| Stage | Component | Algorithm | Time Complexity | Space Complexity |
|-------|-----------|-----------|-----------------|------------------|
| 1 | Extraction | Data Loading | O(n) | O(n) |
| 2 | Preprocessing | Distance Matrix | O(n²) | O(n²) |
| 3.1 | Primary Assignment | Greedy Allocation | O(n) | O(n) |
| 3.2 | Secondary Clustering | K-means | O(n·k·t) | O(n·k) |
| 3.3 | Route Optimization | TSP Algorithms | O(n!) or O(n²) | O(n) |
| 3.4 | Cluster Balancing | Duration Rebalancing | O(n²·R) | O(n) |
| 3.5 | Detailed Routing | OSRM Queries | O(n) | O(n) |
| 4 | Reporting | Metrics Generation | O(n) | O(n) |
| 5 | Loading | File Export | O(n) | O(n) |

Where: n=locations, k=clusters, t=iterations, R=rebalancing iterations

## Quality Metrics

### Optimization Quality
- **Intra-cluster distance**: Average pairwise distance within clusters
- **Duration balance**: Standard deviation of daily workloads
- **Route efficiency**: Drive time vs straight-line distance ratio

### Performance Metrics
- **Total runtime**: End-to-end processing time
- **API call efficiency**: OSRM requests per zone
- **Convergence speed**: Iterations to optimal solution

## System Performance

### Current Implementation
- **Total Runtime**: ~6.36 seconds for 24 zones
- **Concurrency**: 12 worker threads
- **Performance Gain**: 3.58x faster than sequential
- **Optimization Quality**: 67% improved workload balance
- **Scalability**: Linear with zone count

## Assumptions and Limitations

### Key Assumptions
1. **Symmetric TSP**: Drive time A→B equals B→A
2. **Static traffic**: No time-of-day variations
3. **Fixed service times**: Uniform secondary durations
4. **Single vehicle**: No fleet considerations
5. **No precedence**: Any visit order allowed

### Limitations
- No dynamic re-routing capabilities
- No real-time traffic integration
- Single depot assumption
- No multi-day continuity constraints