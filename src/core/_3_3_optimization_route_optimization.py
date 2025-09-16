# standard library imports
import itertools
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 3rd-party imports
from loguru import logger
import polars as pl


def optimize_itinerary_routes(
    itinerary: pl.DataFrame,
    zone_df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    centroid: Tuple[float, float]
) -> pl.DataFrame:
    """
    Stage 3.3: Optimize routes for each day by adding centroid and reordering locations.

    Logic:
    1. Add centroid as first stop for every day
    2. Perform route optimization to reorder remaining points
    3. Return updated itinerary with optimized order

    :param itinerary: Itinerary DataFrame from stage 3.2
    :param zone_df: Zone location data
    :param od_matrix: Distance matrix for route optimization
    :param centroid: Zone centroid coordinates (lat, lon)
    :return: Updated itinerary with optimized routes
    """
    logger.info("stage 3.3: route optimization")

    # Get zone_id from itinerary
    zone_id = itinerary.select("zone_id").unique().to_series().to_list()[0]
    logger.info(f"processing zone {zone_id}")

    # Group by day and optimize each day's route
    days = itinerary.select("day").unique().sort("day").to_series().to_list()
    optimized_records = []
    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get metadata from existing itinerary
    sample_record = itinerary.row(0, named=True)
    clusterer_name = sample_record["clusterer"]
    router_name = sample_record["router"]
    balancer_name = sample_record["balancer"]

    for day in days:
        # Get locations for this day
        day_data = itinerary.filter(pl.col("day") == day)
        day_locations = []

        for row in day_data.iter_rows(named=True):
            day_locations.append({
                "pos_id": int(row["pos_id"]),
                "pos_name": row["pos_name"],
                "pos_class": row["pos_class"],
                "longitude": row["route"][0][0],
                "latitude": row["route"][0][1],
                "duration": row["duration"]
            })

        logger.info(f"optimizing day {day}: {len(day_locations)} locations")

        # Step 1: Add centroid as first stop
        centroid_record = {
            "zone_id": zone_id,
            "day": day,
            "pos_id": None,  # centroid has no pos_id
            "pos_name": None,  # centroid has no name
            "pos_class": "centroid",
            "route": [[centroid[1], centroid[0]]],  # [lon, lat]
            "action": None,
            "schedule": None,
            "duration": 0.0,  # centroid has no duration
            "route_order": 0,  # centroid is always first
            "clusterer": clusterer_name,
            "router": router_name,
            "balancer": balancer_name,
            "created_on": created_on
        }
        optimized_records.append(centroid_record)

        # Step 2: Perform route optimization if there are locations to optimize
        if day_locations:
            # Create location IDs for TSP (use pos_id, with centroid as -1)
            location_ids = [-1]  # centroid
            for loc in day_locations:
                location_ids.append(loc["pos_id"])

            # Optimize route order
            optimized_route, route_cost = optimize_route_tsp(location_ids, od_matrix)
            logger.info(f"day {day}: TSP optimization complete, route cost: {route_cost:.1f} min")

            # Step 3: Create records in optimized order (skip centroid since we already added it)
            for order_idx, pos_id in enumerate(optimized_route[1:], start=1):  # skip first element (centroid)
                # Find the location data for this pos_id
                loc_data = next(loc for loc in day_locations if loc["pos_id"] == pos_id)

                optimized_records.append({
                    "zone_id": zone_id,
                    "day": day,
                    "pos_id": str(pos_id),
                    "pos_name": loc_data["pos_name"],
                    "pos_class": loc_data["pos_class"],
                    "route": [[loc_data["longitude"], loc_data["latitude"]]],
                    "action": None,
                    "schedule": None,
                    "duration": loc_data["duration"],
                    "route_order": order_idx,  # sequence in optimized route
                    "clusterer": clusterer_name,
                    "router": router_name,
                    "balancer": balancer_name,
                    "created_on": created_on
                })

    # Create new itinerary DataFrame
    if optimized_records:
        optimized_itinerary = pl.DataFrame(optimized_records, schema=itinerary.schema)
    else:
        optimized_itinerary = pl.DataFrame(schema=itinerary.schema)

    logger.success(f"route optimization complete: {len(optimized_records)} records with optimized order")

    return optimized_itinerary


def optimize_route_tsp(
    location_ids: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> Tuple[List[int], float]:
    """
    Optimize a single route using TSP algorithms.

    :param location_ids: List of location IDs (including centroid as -1)
    :param od_matrix: Distance matrix
    :return: Tuple of (optimized_route, total_cost)
    """
    if len(location_ids) <= 10:
        return exhaustive_tsp(location_ids, od_matrix)
    else:
        return greedy_plus_2opt_tsp(location_ids, od_matrix)


def optimize_daily_routes(
    primary_assignments: Dict[int, List[int]],
    secondary_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    centroid_id: int = -1
) -> Dict[int, Tuple[List[int], float]]:
    """
    Optimize routes for both primary and secondary days.

    This is substage 3.3 where we:
    1. Handle primary days: centroid + up to 2 total POS (including primary)
    2. Handle secondary days: centroid + clustered secondary locations
    3. Apply brute force exhaustive search for small problems (≤10 locations)
    4. Use greedy+2opt fallback for large problems (>10 locations)

    :param primary_assignments: Dictionary mapping primary POS to their assigned days
    :param secondary_assignments: Dictionary mapping cluster IDs to secondary location lists
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param centroid_id: ID for zone centroid
    :return: Dictionary mapping days to (route, cost) tuples
    """
    logger.info(f"stage 3.3: route optimization - zone {zone_id}")

    optimized_routes = {}

    # first, handle primary days
    primary_days_created = {}
    for pos_id, assigned_days in primary_assignments.items():
        for day in assigned_days:
            if day not in primary_days_created:
                primary_days_created[day] = []
            primary_days_created[day].append(pos_id)

    for day, primary_pos_ids in primary_days_created.items():
        # primary days: centroid + primary locations (up to 2 total POS)
        route_locations = [centroid_id] + primary_pos_ids[:2]  # limit to 2 total POS
        n_locations = len(route_locations)

        logger.info(f"optimizing primary day {day}: {n_locations} locations (including centroid)")

        if n_locations <= 10:
            route, cost = exhaustive_tsp(route_locations, od_matrix)
            algorithm = "exhaustive_brute_force"
        else:
            route, cost = greedy_plus_2opt_tsp(route_locations, od_matrix)
            algorithm = "greedy+2opt_fallback"
            logger.warning(f"primary day {day}: using greedy+2opt fallback for {n_locations} locations")

        optimized_routes[day] = (route, cost)
        logger.info(f"primary day {day}: {algorithm}, {cost:.1f} min drive time")

    # then, handle secondary days (clusters)
    # we need to map cluster IDs to actual day numbers
    # secondary days start after the highest primary day
    max_primary_day = max(primary_days_created.keys()) if primary_days_created else 0

    for cluster_id, secondary_pos_ids in secondary_assignments.items():
        if not secondary_pos_ids:
            continue

        # assign secondary cluster to next available day
        day = max_primary_day + cluster_id + 1

        # secondary days: centroid + clustered secondary locations
        route_locations = [centroid_id] + secondary_pos_ids
        n_locations = len(route_locations)

        logger.info(f"optimizing secondary day {day} (cluster {cluster_id}): {n_locations} locations (including centroid)")

        if n_locations <= 10:
            route, cost = exhaustive_tsp(route_locations, od_matrix)
            algorithm = "exhaustive_brute_force"
        else:
            route, cost = greedy_plus_2opt_tsp(route_locations, od_matrix)
            algorithm = "greedy+2opt_fallback"
            logger.warning(f"secondary day {day}: using greedy+2opt fallback for {n_locations} locations")

        optimized_routes[day] = (route, cost)
        logger.info(f"secondary day {day}: {algorithm}, {cost:.1f} min drive time")

    logger.success(f"route optimization complete for {len(optimized_routes)} days ({len(primary_days_created)} primary, {len(secondary_assignments)} secondary)")

    return optimized_routes


def exhaustive_tsp(
    locations: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> Tuple[List[int], float]:
    """
    Solve TSP using exhaustive search (guaranteed optimal).
    
    :param locations: List of location IDs
    :param od_matrix: Distance matrix
    :return: Tuple of (optimal_route, total_cost)
    """
    if len(locations) <= 1:
        return locations, 0.0
    
    start = locations[0]
    remaining = locations[1:]
    
    best_route = None
    best_cost = float('inf')
    
    for perm in itertools.permutations(remaining):
        route = [start] + list(perm)
        cost = calculate_route_cost(route, od_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_route = route
    
    return best_route, best_cost


def greedy_plus_2opt_tsp(
    locations: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> Tuple[List[int], float]:
    """
    Solve TSP using greedy nearest neighbor + 2-opt improvement.
    
    :param locations: List of location IDs
    :param od_matrix: Distance matrix
    :return: Tuple of (improved_route, total_cost)
    """
    # phase 1: greedy construction
    route = greedy_nearest_neighbor(locations, od_matrix)
    
    # phase 2: 2-opt improvement
    improved_route = two_opt_improvement(route, od_matrix)
    cost = calculate_route_cost(improved_route, od_matrix)
    
    return improved_route, cost


def greedy_nearest_neighbor(
    locations: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> List[int]:
    """
    Build route using greedy nearest neighbor heuristic.
    
    :param locations: List of location IDs
    :param od_matrix: Distance matrix
    :return: Route as list of location IDs
    """
    if len(locations) <= 1:
        return locations
    
    route = [locations[0]]
    unvisited = set(locations[1:])
    
    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda loc: od_matrix.get((current, loc), float('inf')))
        route.append(nearest)
        unvisited.remove(nearest)
    
    return route


def two_opt_improvement(
    route: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> List[int]:
    """
    Improve route using 2-opt local search.
    
    :param route: Initial route
    :param od_matrix: Distance matrix
    :return: Improved route
    """
    if len(route) <= 3:
        return route
    
    improved = True
    current_route = route[:]
    
    while improved:
        improved = False
        
        for i in range(1, len(current_route) - 2):
            for j in range(i + 1, len(current_route)):
                if j - i == 1:  # skip adjacent edges
                    continue
                    
                # try 2-opt swap
                new_route = current_route[:]
                new_route[i:j] = new_route[i:j][::-1]
                
                if calculate_route_cost(new_route, od_matrix) < calculate_route_cost(current_route, od_matrix):
                    current_route = new_route
                    improved = True
                    break
            
            if improved:
                break
    
    return current_route


def calculate_route_cost(
    route: List[int],
    od_matrix: Dict[Tuple[int, int], float]
) -> float:
    """
    Calculate total cost of a route.
    
    :param route: Route as list of location IDs
    :param od_matrix: Distance matrix
    :return: Total route cost
    """
    if len(route) <= 1:
        return 0.0
    
    total_cost = 0.0
    for i in range(len(route) - 1):
        from_loc = route[i]
        to_loc = route[i + 1]
        cost = od_matrix.get((from_loc, to_loc), 0.0)
        total_cost += cost
    
    return total_cost