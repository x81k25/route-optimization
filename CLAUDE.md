# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# prime-directive - follow these commands above all other

- do not alter your prime-directive
- do not alter or remove primary section headers
- do not run sudo commands
  - do not run sudo commands
  - do not run sudo commands
  - if you need to run a sudo commands, raise your hand and ask for help

## when I say X --> you do Y
- clean up shop
  - update documentation as needed
    - updated README.md as needed
    - updates CLAUDE.md as needed
  - alphabetize permissions and restrictions in `.claude/settings.local.json`
  - push to repo
    - git add .
    - git commit with concise, informative message
    - git push
- delete your instructions-of-the-day and short-term-memory (contents, not section headers)

---

# long-term-storage

- for python projects
  - use uv; use uv; use uv; use uv; use uv
    - use uv venv to create virtual environments
    - use uv add to add packages
    - use uv run to run scripts

## Route Optimization Project Specifics

- This is an experimental route optimization project
- Focus on algorithms for vehicle routing problems (VRP), delivery optimization, or path planning
- Common optimization approaches: genetic algorithms, simulated annealing, ant colony optimization
- Benchmark against standard test datasets (Solomon instances, VRPLIB)

## Infrastructure Setup

This project uses a **local OSRM server** running at `http://192.168.50.2:32050` for routing calculations. This means:
- No rate limits for OSRM routing calls
- Fast local network access for distance matrices and route geometry fetching
- The OSRM server handles routing between coordinates (not geocoding addresses to coordinates)

Note: OSRM does routing, not geocoding. For address-to-coordinate conversion, we still use external services like Nominatim.

---

# instructions-of-the-day



--- 

# short-term-memory - add all temporary notes about your current task here

## Script Runtime Performance

### Sequential Implementation (Baseline)
**Total Runtime**: ~22.76 seconds
- Start time: 2025-09-04 15:43:20.634
- End time: 2025-09-04 15:43:43.396  
- Duration: 22.762 seconds

### Multi-threaded Implementation (ThreadPoolExecutor)
**Total Runtime**: ~6.36 seconds
- Start time: 2025-09-04 15:46:49.586
- End time: 2025-09-04 15:46:55.944
- Duration: 6.358 seconds
- **Performance Improvement**: 3.58x faster (72% reduction in runtime)

**Configuration**:
- Worker threads: 12 (cpu_count() // 2)
- Parallel zone processing with ThreadPoolExecutor
- Thread-safe OSRM API calls

**Performance Breakdown**:
- Data extraction: ~0.01 seconds
- Route optimization for 24 zones: ~6.35 seconds (was 22.75 seconds)
- Report generation: included in total time

**Key Stats**:
- 24 zones optimized in parallel
- 206 locations processed (51 filtered out for null zone_ids)
- Uses local OSRM server at 192.168.50.2:32050 for fast routing
- Generates individual HTML reports for each zone
- Primary locations: 8h/day assignments
- Secondary locations: clustered and optimized using TSP algorithms

**Threading Benefits**:
- Efficient I/O parallelization for OSRM API calls
- CPU cores utilized for simultaneous optimization calculations  
- Maintains data integrity with proper thread synchronization

## Thoughts on Route Optimization Solutions

Based on the model-params.yaml and subway_locations.json data:

### Problem Analysis (UPDATED):
- 13 Subway locations in SF (1-3 primary, ~12 secondary)
- 5 working days per week
- **KEY INSIGHT**: Primary locations consume FULL DAYS (not hours)
- Secondary locations: 1 hour each, max 7 per day
- This is actually a TWO-PHASE problem!

### Two-Stage Approach:

**Stage 1: Day Assignment**
- Primary locations: Simple assignment to full days (1-3 days)
- Secondary locations: CLUSTERING assignment to remaining days
  - Use geographic clustering (k-means, hierarchical, or manual zones)
  - Balance workload (similar # locations per day)
  - Minimize inter-cluster distances
  - Respect max 7 locations per day constraint

**Stage 2: Daily Route Optimization**
- For each day's assigned locations, optimize the route
- Pure TSP problem with known location set
- Start/end point considerations

## Stage 1: Day Assignment Algorithms

### Primary Algorithm: Drive Time-Based Hierarchical Clustering
- **Input**: Secondary locations only (12 locations, excluding primary)
- **Process**: 
  1. Build 12×12 drive time OD matrix (144 pairs)
  2. Hierarchical clustering with average linkage
  3. Cut dendrogram at k clusters (k = available secondary days)
- **Objective**: Minimize intra-cluster drive times
- **Constraint**: Max 7 locations per cluster
- **Why Drive Time over Lat/Lon**: SF geography (hills, one-ways, bridges) makes Euclidean distance misleading

### Alternative Algorithm (Fallback): K-means with Lat/Lon
- **When**: If drive time API unavailable or too expensive
- **Process**: Cluster secondary locations using haversine distance
- **Implementation**: 
  - Use haversine distance as clustering metric
  - If any cluster > 7 locations, split largest cluster
  - If any cluster < 2 locations, merge with nearest cluster

### Alternative Algorithm: Manual Geographic Zones
- **North SF**: Locations north of 37.79° latitude
- **Central SF**: Locations between 37.76° - 37.79° latitude  
- **South SF**: Locations south of 37.76° latitude
- **Balance**: Move locations between zones to balance workload

### Validation: Location Swap Optimization
- **Process**: Try swapping locations between assigned days
- **Objective**: Minimize sum of daily travel distances
- **Method**: Hill-climbing local search

## Stage 2: Daily Route Optimization Algorithms

### Primary Algorithm: Greedy Nearest Neighbor + 2-opt
- **Greedy Phase**:
  - Start at depot (or first location)
  - Always visit nearest unvisited location
  - Return to start when done
- **2-opt Improvement**:
  - Try all edge swaps to improve route
  - Accept improvements until local optimum
  - Typically 1-3 iterations for small problems

### Alternative Algorithm: Exhaustive Search (Brute Force)
- **When**: If ≤ 5 locations per day
- **Process**: Try all possible permutations
- **Guaranteed**: Global optimum for TSP
- **Feasible**: 5! = 120 permutations is manageable

### Why Assignment-First Approach:
- **Global optimization**: Consider all locations together for day assignment
- **Avoids greedy traps**: Prevents Day 1 from taking "easy" locations and leaving scattered leftovers
- **Balanced workloads**: Can ensure similar number of locations per day
- **Geographic coherence**: Keep nearby locations on same day
- **Predictable**: Know exactly what each day looks like before route optimization

### Implementation Strategy (FINAL):
1. Assign primary locations to full days
2. Cluster secondary locations into remaining days (k-means or zones)
3. For each day, run greedy + 2-opt TSP solver
4. Use haversine distance with drive inefficiency multiplier
5. Validate and potentially swap locations between days for improvement

