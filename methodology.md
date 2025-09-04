# Route Optimization Methodology

This document outlines the data science methodologies, algorithms, and mathematical assumptions underlying the route optimization system.

## Problem Formulation

### Two-Stage Hierarchical Optimization
The system addresses route optimization as a **two-stage hierarchical problem**:

1. **Stage 1: Day Assignment Problem** - Assigning locations to specific workdays
2. **Stage 2: Traveling Salesman Problem (TSP)** - Optimizing route order within each day

This decomposition allows for tractable solutions to the Vehicle Routing Problem (VRP) variant with time windows and capacity constraints.

### Core Assumptions

#### Location Classification
- **Primary Locations**: High-value locations requiring extended time investment
- **Secondary Locations**: Standard locations with fixed service times

#### Time Allocation Model
- Primary locations consume full days based on weekly hour allocation
- Secondary locations have fixed service duration (1 hour default)
- Work capacity constraint: maximum 7 locations per day

## Stage 1: Day Assignment Algorithms

### Primary Location Assignment
**Algorithm**: Greedy Time-Based Assignment
```
Hours per primary = total_primary_hours / num_primary_locations
For each primary location:
    Distribute hours across available days
    Fill days sequentially until all hours allocated
```

**Mathematical Foundation**:
- Time allocation: `h_p = H_total / |P|` where `H_total` = weekly primary hours, `|P|` = primary locations
- Day utilization: Fill days to capacity before moving to next day

### Secondary Location Clustering
**Algorithm**: Hierarchical Agglomerative Clustering with Drive-Time Distance Matrix

**Distance Metric**: Drive time (minutes) between all location pairs
```python
distance_matrix[i,j] = drive_time(location_i, location_j)
```

**Clustering Method**: 
- Linkage: Average linkage (UPGMA)
- Criterion: Cut dendrogram at k clusters where k = available secondary days
- Constraint handling: Split clusters exceeding 7 locations

**Mathematical Formulation**:
```
minimize: Σ(intra_cluster_distances)
subject to: |cluster_i| ≤ 7 ∀i
```

**Constraint Enforcement**: `_enforce_cluster_size_constraints()` splits oversized clusters using simple partitioning.

## Stage 2: Daily Route Optimization (TSP)

### Algorithm Selection Strategy
**Adaptive algorithm selection based on problem size**:
- **Small problems (≤5 locations)**: Exhaustive search (guaranteed optimal)
- **Large problems (>5 locations)**: Greedy Nearest Neighbor + 2-opt improvement

### Exhaustive Search Algorithm
**Method**: Brute force enumeration with fixed starting point
```python
for permutation in itertools.permutations(remaining_locations):
    route = [fixed_start] + list(permutation)
    evaluate route_cost
    keep best_route
```

**Complexity**: O(n!) where n = number of locations
**Guarantee**: Global optimum for TSP

### Greedy + 2-opt Algorithm

#### Phase 1: Greedy Nearest Neighbor Construction
**Algorithm**: 
```
current = start_location
while unvisited:
    next = argmin(drive_time(current, location)) for location in unvisited
    add next to route
    current = next
```

**Complexity**: O(n²)
**Quality**: Approximation with no guaranteed bound

#### Phase 2: 2-opt Local Improvement
**Algorithm**: 
```
for i in range(1, n-2):
    for j in range(i+1, n):
        new_route = reverse_segment(route, i, j)
        if cost(new_route) < cost(current_route):
            current_route = new_route
```

**Mathematical Operation**: 
For route segment reversal between positions i and j:
```
route[i:j+1] = route[i:j+1][::-1]
```

**Termination**: Local optimum (no improving 2-opt moves remain)
**Complexity**: O(n²) per iteration

## Geographic Clustering (Zone Creation)

### K-means Spatial Clustering
**Objective Function**:
```
minimize: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```
where `μᵢ` = centroid of cluster i, `Cᵢ` = cluster i

**Distance Metric**: Haversine distance (great-circle distance)
```python
haversine_distance = R * 2 * arcsin(sqrt(
    sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
))
```
where R = 6371 km (Earth's radius)

**Initialization**: K-means++ initialization (10 random initializations, best result selected)

### Cluster Balancing Algorithm
**Constraint Satisfaction**: Ensure cluster sizes within [min_size, max_size] bounds

**Method**: 
1. Identify violating clusters (oversized/undersized)
2. Extract locations from violating clusters  
3. Re-cluster extracted locations using K-means
4. Merge with compliant clusters

**Rebalancing Strategy**: Target size = (min_size + max_size) / 2

## Mathematical Constraints & Parameters

### Model Parameters
```yaml
days_per_week: 5                 # Available working days
hours_per_day: 8                # Daily capacity
primary_hours_per_week: 24      # Total weekly primary hours
hours_per_non_primary: 1        # Service time per secondary location
locations_per_day_max: 7        # Maximum locations per day
```

### Constraint Formulation
1. **Capacity Constraint**: `Σ(locations_in_day) ≤ 7`
2. **Time Constraint**: `Σ(service_times) + Σ(drive_times) ≤ 8 hours`
3. **Primary Hours Constraint**: `Σ(primary_hours) = 24 hours/week`
4. **Assignment Constraint**: Each location assigned to exactly one day

## Distance and Cost Functions

### Drive Time Calculation
**Source**: OSRM (Open Source Routing Machine) local server
**Metric**: Minutes of driving time between coordinate pairs
**Matrix Construction**: All-pairs shortest path using Dijkstra's algorithm

### Route Cost Function
```python
total_cost = Σᵢ drive_time(location[i], location[i+1])
```

No additional penalties or weights applied - pure drive time minimization.

## Algorithmic Complexity Analysis

| Component | Algorithm | Time Complexity | Space Complexity |
|-----------|-----------|----------------|-----------------|
| Zone Clustering | K-means | O(n·k·d·t) | O(n·d) |
| Day Assignment | Hierarchical Clustering | O(n³) | O(n²) |
| Small TSP | Exhaustive Search | O(n!) | O(n) |
| Large TSP | Greedy + 2-opt | O(n²·I) | O(n) |

Where: n=locations, k=clusters, d=dimensions, t=iterations, I=2-opt iterations

## Quality Metrics & Evaluation

### Clustering Quality
- **Intra-cluster distance**: Average pairwise distance within clusters
- **Cluster size balance**: Standard deviation of cluster sizes
- **Silhouette coefficient**: Not implemented (could be added)

### Route Quality  
- **Total drive time**: Sum of inter-location drive times
- **Route efficiency**: Drive time / straight-line distance ratio
- **Daily utilization**: Service hours + drive hours per day

### Performance Tracking
The system logs optimization metadata:
- Algorithm selection rationale
- Convergence iterations (for 2-opt)
- Solution quality indicators (optimal vs heuristic)

## Assumptions and Limitations

### Key Assumptions
1. **Symmetric TSP**: Drive time from A→B equals B→A
2. **No traffic variation**: Static drive times throughout day
3. **Fixed service times**: All secondary locations require same duration
4. **Single vehicle**: No fleet considerations
5. **No precedence constraints**: Locations can be visited in any order

### Known Limitations
1. **Local optima**: 2-opt may converge to suboptimal solutions
2. **Clustering myopia**: Stage 1 clustering ignores Stage 2 TSP costs
3. **No dynamic rebalancing**: Day assignments are static once set
4. **Limited constraint handling**: Complex business rules not modeled

### Future Enhancements
- **Integrated optimization**: Joint day-assignment and routing optimization
- **Metaheuristics**: Genetic algorithms, simulated annealing for larger instances
- **Dynamic updates**: Real-time reoptimization based on traffic/delays
- **Multi-objective optimization**: Balance drive time vs service quality metrics

## Variations

### Variation 1 (Baseline)
**Clustering Algorithm**: Hierarchical Agglomerative Clustering with Drive-Time Distance Matrix  
**Routing Algorithm**: Adaptive (Exhaustive ≤5 locations, Greedy+2-opt >5 locations)  
**Model Assumptions**: Standard (7 locations/day max, 1hr secondary locations, 24hr primary/week)  
**Results**:
- Average daily drive time: 1.96 minutes
- Average weekly duration: 24.04 hours  
- Average utilization: 60.10%
- Average overutilized days: 0.54
- Average underutilized days: 2.42