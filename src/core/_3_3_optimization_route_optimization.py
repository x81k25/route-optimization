"""
Stage 3.3: Route Optimization
Optimize visit sequences within each day using TSP algorithms
"""

# standard library imports
import itertools
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl


def optimize_daily_routes(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    centroid_id: int = -1
) -> Dict[int, Tuple[List[int], float]]:
    """
    Optimize routes for each day's assigned locations.
    
    This is substage 3.3 where we:
    1. Apply brute force exhaustive search for small problems (≤10 locations)
    2. Use greedy+2opt fallback for large problems (>10 locations)
    3. Prevents computational intractability from factorial complexity
    
    :param day_assignments: Dictionary mapping days to location lists
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param centroid_id: ID for zone centroid
    :return: Dictionary mapping days to (route, cost) tuples
    """
    logger.info(f"Stage 3.3: ROUTE OPTIMIZATION - Zone {zone_id}")
    
    optimized_routes = {}
    
    for day, location_ids in day_assignments.items():
        if not location_ids:
            continue
            
        # add centroid as starting point
        route_locations = [centroid_id] + location_ids
        n_locations = len(route_locations)
        
        logger.info(f"Optimizing day {day}: {n_locations} locations (including centroid)")
        
        # use brute force exhaustive search, with fallback for large problems
        if n_locations <= 10:
            route, cost = exhaustive_tsp(route_locations, od_matrix)
            algorithm = "exhaustive_brute_force"
        else:
            route, cost = greedy_plus_2opt_tsp(route_locations, od_matrix)
            algorithm = "greedy+2opt_fallback"
            logger.warning(f"Day {day}: Using greedy+2opt fallback for {n_locations} locations (>10 threshold)")
        
        optimized_routes[day] = (route, cost)
        logger.info(f"Day {day}: {algorithm}, {cost:.1f} min drive time")
    
    logger.success(f"Route optimization complete for {len(optimized_routes)} days")
    
    return optimized_routes


def exhaustive_tsp(locations: List[int], od_matrix: Dict[Tuple[int, int], float]) -> Tuple[List[int], float]:
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


def greedy_plus_2opt_tsp(locations: List[int], od_matrix: Dict[Tuple[int, int], float]) -> Tuple[List[int], float]:
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


def greedy_nearest_neighbor(locations: List[int], od_matrix: Dict[Tuple[int, int], float]) -> List[int]:
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


def two_opt_improvement(route: List[int], od_matrix: Dict[Tuple[int, int], float]) -> List[int]:
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


def calculate_route_cost(route: List[int], od_matrix: Dict[Tuple[int, int], float]) -> float:
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