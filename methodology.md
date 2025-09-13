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

The system supports multiple clustering algorithms for grouping secondary locations into daily routes. The following comparison table shows the available approaches:

# Clustering Models Comparison for Multi-Day TSP

| Model | Input Requirements | Key Parameters | Strengths | Limitations | Implementation Complexity | Balance Control | Fixed Day Count Adaptability | Handles Irregular Clusters | Computational Efficiency | Output Format | Stability | Interpretability | Sensitivity to Outliers | Adaptation to Constraints |
|-------|-------------------|----------------|-----------|-------------|--------------------------|----------------|----------------------------|----------------------------|-------------------------|---------------|-----------|-----------------|------------------------|---------------------------|
| MDS+k-means with drive times | Complete drive time matrix | • n_clusters (required): Number of days<br>• random_state: For reproducibility<br>• n_init: Number of restarts<br>• MDS n_components: Usually 2 | • Preserves global drive time relationships<br>• Works well with Euclidean and non-Euclidean distances<br>• Finds compact, evenly sized clusters | • Can get stuck in local optima<br>• Assumes roughly spherical clusters<br>• Sensitive to initialization | Medium | Medium<br>(Can be improved with constrained variants) | High<br>(Directly takes number of days as input) | Low<br>(Prefers convex, circular clusters) | High<br>(O(n²) for MDS, O(kni) for k-means) | Array of day assignments for each location | Medium<br>(Results may vary with different initializations) | High<br>(Intuitive centroids and assignments) | Medium<br>(Outliers can significantly affect cluster shapes) | Medium<br>(Constraints can be added but requires customization) |
| DBSCAN on MDS-transformed drive times | Complete drive time matrix | • eps (required): Maximum distance between points in neighborhood<br>• min_samples (required): Minimum points to form cluster<br>• MDS n_components: Usually 2 | • Discovers clusters of arbitrary shapes<br>• Identifies outliers<br>• No assumptions about cluster shape | • Struggles with varying density clusters<br>• Difficult to tune parameters correctly<br>• May create "noise" points not in any cluster | High | Low<br>(No inherent balancing mechanism) | Low<br>(Cannot directly control number of clusters) | High<br>(Excellent at finding irregular shapes) | Medium<br>(O(n²) for MDS, O(n²) for DBSCAN) | Array of day assignments with possible -1 for outliers | High<br>(Deterministic results) | Medium<br>(Clusters intuitive but parameters less so) | Low<br>(Explicitly identifies outliers) | Low<br>(Difficult to add constraints) |
| Hierarchical clustering on drive time matrix | Complete drive time matrix | • n_clusters (required): Number of days<br>• linkage (required): 'single', 'complete', 'average', or 'ward'<br>• distance_threshold: Alternative to n_clusters | • Creates intuitive dendrogram visualization<br>• Works directly with drive time distances<br>• Multiple clustering levels available | • Can be sensitive to small perturbations<br>• Different linkage methods produce very different results<br>• May create imbalanced clusters | Medium | Low<br>(No inherent balancing mechanism) | High<br>(Can cut dendrogram at desired level) | Medium<br>(Depends on linkage method) | Medium<br>(O(n²) for distance calculations, O(n³) for some linkage methods) | Hierarchical structure plus array of cluster assignments | High<br>(Deterministic results) | Very High<br>(Dendrogram shows relationships clearly) | Medium<br>(Depends on linkage method used) | Medium<br>(Can incorporate constraints via custom distance metrics) |
| Spectral clustering on drive time matrix | Complete drive time matrix | • n_clusters (required): Number of days<br>• affinity (required): 'precomputed' for drive time matrix<br>• n_init: Number of runs<br>• eigen_solver: Method for eigen decomposition | • Excellent for complex, non-convex clusters<br>• Handles data with complex relationships<br>• Based on graph theory approach | • Parameter tuning can be challenging<br>• Computationally more expensive<br>• Sensitive to choice of similarity matrix | High | Low<br>(No inherent balancing mechanism) | High<br>(Takes number of clusters as input) | Very High<br>(Excellent at finding complex cluster shapes) | Low<br>(O(n³) in worst case for eigen decomposition) | Array of day assignments for each location | Medium<br>(Results may vary with different runs) | Low<br>(Complex mathematical foundation) | Low<br>(Generally robust to outliers) | Medium<br>(Can incorporate constraints but complex) |
| Balanced partitioning with drive times | Complete drive time matrix<br>Optional: stop duration/importance | • n_clusters (required): Number of days<br>• balance_param (required): Weight of balance constraint<br>• max_iter: Maximum iterations<br>• size_min/size_max: Min/max cluster sizes | • Explicitly controls cluster sizes<br>• Can enforce minimum/maximum stops per day<br>• Optimizes for both proximity and balance | • Complex to implement from scratch<br>• May sacrifice some proximity for balance<br>• Often requires custom implementation | Very High | Very High<br>(Primary purpose is balancing) | High<br>(Takes number of clusters as input) | Low<br>(Typically produces convex clusters) | Medium<br>(Depends on implementation, generally O(kni)) | Array of day assignments with balanced cluster sizes | Medium<br>(Depends on implementation) | High<br>(Balance constraints are easy to understand) | Medium<br>(Outliers impact balance objectives) | Very High<br>(Designed specifically for constraints) |

**Default Algorithm**: MDS+k-means with drive times

**Distance Metric**: Haversine distance (great-circle)
```python
haversine_distance = R * 2 * arcsin(sqrt(
    sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
))
```
where R = 6371 km (Earth's radius)

**Noise Detection**:
- Isolation threshold: 150km radius
- Minimum neighbors: 2 locations
- Excluded points: Assigned zone_id: null

#### 3.3 Route Optimization

**Algorithm Selection**:
- **Brute Force Exhaustive Search** (Default): Complete enumeration of all possible routes
- **Adaptive Algorithm Selection**: Exhaustive search for small problems (≤5 locations), Greedy + 2-opt for larger problems

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

The system supports multiple balancing approaches for equalizing workloads across daily clusters. The following comparison table shows the available methods:

# Balancing Approaches Comparison for Multi-Day TSP

| Approach | Input Requirements | Key Parameters | Strengths | Limitations | Implementation Complexity | Computational Efficiency | Stability | Optimality Guarantee | Adaptability to Constraints | Quality of Balance |
|----------|-------------------|----------------|-----------|-------------|--------------------------|-------------------------|-----------|---------------------|----------------------------|-------------------|
| Enhanced Greedy Transfer | • Clustered stops<br>• Drive time matrix<br>• Current route times per day | • threshold: Min time difference to trigger balancing<br>• max_iterations: Iteration limit<br>• selection_criteria: How to choose stops to transfer | • Simple implementation<br>• Intuitive approach<br>• Directly addresses imbalance<br>• Fast execution | • Can get stuck in local optima<br>• May make suboptimal transfers<br>• No backtracking capability<br>• Sequence-dependent results | Low | Very High<br>(O(n²) per iteration, few iterations) | Medium<br>(Results may depend on initial clusters) | Low<br>(No guarantee of global optimality) | High<br>(Easy to add constraints to transfer criteria) | Medium<br>(Improves balance but may miss optimal transfers) |
| Local Search with Swap Operations | • Clustered stops<br>• Drive time matrix<br>• Current route times per day | • threshold: Min improvement to accept a swap<br>• max_iterations: Iteration limit<br>• neighborhood_size: How many swaps to consider<br>• tabu_list_size: For tabu search variants | • Explores larger solution space<br>• Can escape some local optima<br>• Considers pairwise interactions<br>• Effective for moderately complex problems | • More complex implementation<br>• Higher computational requirements<br>• May still miss global optima<br>• Requires careful parameter tuning | Medium | High<br>(O(n²m²) where m is clusters, typically fast) | Medium-High<br>(Less dependent on initial clusters) | Low<br>(Better than greedy but no guarantees) | Medium<br>(Constraints add complexity to swap evaluations) | High<br>(Effective at improving balance) |
| Simulated Annealing | • Clustered stops<br>• Drive time matrix<br>• Current route times per day | • initial_temperature: Starting temperature<br>• cooling_rate: How quickly temp decreases<br>• min_temperature: Stopping condition<br>• iterations_per_temp: Iterations at each temp<br>• move_operators: Types of moves allowed | • Can escape local optima<br>• Proven effectiveness for complex problems<br>• Probabilistic acceptance of uphill moves<br>• Theoretical convergence properties | • Parameter tuning can be challenging<br>• Stochastic results<br>• Longer runtime<br>• No optimality guarantee without infinite time | High | Medium<br>(O(n²) per iteration, many iterations) | Low<br>(Results vary between runs)<br>(Can be increased with multiple runs) | Medium<br>(Can approach optimal solutions with sufficient cooling) | Medium<br>(Constraints can be included in move evaluation) | Very High<br>(When properly tuned, finds excellent balance) |
| Min-Max Optimization | • Clustered stops<br>• Drive time matrix<br>• Current route times per day | • threshold: Acceptable difference between max/min day<br>• max_iterations: Iteration limit<br>• step_size: How many stops to move per iteration<br>• focus_factor: Emphasis on min vs max improvement | • Directly targets the extremes<br>• Clear focus on minimizing deviation<br>• Mathematically sound approach<br>• Effective for balance-critical applications | • May over-optimize extremes at the expense of average performance<br>• Can oscillate between solutions<br>• Less flexible than some approaches | Medium | High<br>(O(nm) where m is clusters, typically fast) | High<br>(Deterministic process with consistent results) | Medium<br>(Guarantees minimizing maximum difference under certain conditions) | Medium<br>(Constraints can be incorporated into the min/max calculations) | Very High<br>(Explicitly designed to optimize balance) |
| Network Flow Formulation | • Clustered stops<br>• Drive time matrix<br>• Drive time impact of each stop on each day | • cost_function: How to value assignments<br>• flow_capacity: Limits on assignments per day<br>• node_weights: Importance of each stop<br>• solver_parameters: For the network flow algorithm | • Strong theoretical foundation<br>• Can find globally optimal transfers<br>• Handles complex relationships<br>• Polynomial time complexity | • Most complex implementation<br>• Requires specialized solvers/libraries<br>• Abstract formulation<br>• May be overkill for simpler instances | Very High | Medium<br>(O(n³) in worst case, but typically faster with specialized algorithms) | Very High<br>(Deterministic with consistent results) | High<br>(Can guarantee optimal transfers under the model constraints) | High<br>(Natural framework for adding various constraints) | Very High<br>(Can achieve optimal balance within the model constraints) |

**Default Algorithm**: Enhanced Greedy Transfer

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