"""
Stage 2: Daily Route Optimization

Optimizes the route for a given set of locations using Greedy Nearest Neighbor + 2-opt.
"""

import pickle
import numpy as np
import polars as pl
from typing import List, Tuple, Dict
import itertools


class DailyRouteOptimizer:
    """Optimizes routes for a single day's locations."""
    
    def __init__(self, od_matrix_path: str = "data/od_matrix.pkl"):
        """
        Initialize with OD matrix.
        
        Args:
            od_matrix_path: Path to pickled polars DataFrame with drive times
        """
        with open(od_matrix_path, 'rb') as f:
            self.od_matrix = pickle.load(f)
        
        # Create lookup dictionary for quick drive time access
        self.drive_time_lookup = {}
        for row in self.od_matrix.iter_rows():
            origin_id, dest_id, drive_time = row[0], row[1], row[5]
            self.drive_time_lookup[(origin_id, dest_id)] = drive_time
    
    def get_drive_time(self, origin_id: int, dest_id: int) -> float:
        """Get drive time between two locations."""
        if origin_id == dest_id:
            return 0.0
        return self.drive_time_lookup.get((origin_id, dest_id), float('inf'))
    
    def greedy_nearest_neighbor(self, location_ids: List[int], start_location_id: int = None) -> Tuple[List[int], float]:
        """
        Solve TSP using Greedy Nearest Neighbor algorithm.
        
        Args:
            location_ids: List of location IDs to visit
            start_location_id: Starting location (if None, uses first location)
            
        Returns:
            Tuple of (route, total_drive_time)
        """
        if len(location_ids) <= 1:
            return location_ids, 0.0
        
        # Choose starting location
        current_location = start_location_id if start_location_id is not None else location_ids[0]
        unvisited = set(location_ids) - {current_location}
        route = [current_location]
        total_time = 0.0
        
        # Greedy selection: always visit nearest unvisited location
        while unvisited:
            nearest_location = min(
                unvisited,
                key=lambda loc: self.get_drive_time(current_location, loc)
            )
            
            drive_time = self.get_drive_time(current_location, nearest_location)
            total_time += drive_time
            
            route.append(nearest_location)
            unvisited.remove(nearest_location)
            current_location = nearest_location
        
        return route, total_time
    
    def two_opt_improvement(self, route: List[int], max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        Improve route using 2-opt local search.
        
        Args:
            route: Initial route as list of location IDs
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Tuple of (improved_route, total_drive_time)
        """
        if len(route) <= 3:
            return route, self._calculate_route_time(route)
        
        best_route = route.copy()
        best_time = self._calculate_route_time(best_route)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try all possible 2-opt swaps
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:  # Skip adjacent edges
                        continue
                    
                    # Create new route by reversing segment between i and j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_time = self._calculate_route_time(new_route)
                    
                    if new_time < best_time:
                        best_route = new_route.copy()
                        best_time = new_time
                        improved = True
            
            if improved:
                route = best_route.copy()
            else:
                break  # No more improvements found
        
        return best_route, best_time
    
    def _calculate_route_time(self, route: List[int]) -> float:
        """Calculate total drive time for a route."""
        if len(route) <= 1:
            return 0.0
        
        total_time = 0.0
        for i in range(len(route) - 1):
            total_time += self.get_drive_time(route[i], route[i + 1])
        
        return total_time
    
    def exhaustive_search(self, location_ids: List[int]) -> Tuple[List[int], float]:
        """
        Solve TSP using exhaustive search (brute force).
        Only use for small problems (≤ 8 locations).
        
        Args:
            location_ids: List of location IDs to visit
            
        Returns:
            Tuple of (optimal_route, total_drive_time)
        """
        if len(location_ids) > 8:
            raise ValueError("Exhaustive search only supported for ≤ 8 locations")
        
        if len(location_ids) <= 1:
            return location_ids, 0.0
        
        best_route = None
        best_time = float('inf')
        
        # Fix first location, permute the rest
        first_location = location_ids[0]
        remaining_locations = location_ids[1:]
        
        for perm in itertools.permutations(remaining_locations):
            route = [first_location] + list(perm)
            route_time = self._calculate_route_time(route)
            
            if route_time < best_time:
                best_route = route
                best_time = route_time
        
        return best_route, best_time
    
    def optimize_route(
        self, 
        location_ids: List[int], 
        start_location_id: int = None,
        use_exhaustive_if_small: bool = True
    ) -> Tuple[List[int], float, Dict[str, any]]:
        """
        Main method to optimize a route for given locations.
        
        Args:
            location_ids: List of location IDs to visit
            start_location_id: Starting location (if None, uses first location)
            use_exhaustive_if_small: Use exhaustive search for ≤ 5 locations
            
        Returns:
            Tuple of (route, total_time, metadata)
        """
        if not location_ids:
            return [], 0.0, {'algorithm': 'empty'}
        
        metadata = {
            'n_locations': len(location_ids),
            'start_location': start_location_id or location_ids[0]
        }
        
        # Choose algorithm based on problem size
        if use_exhaustive_if_small and len(location_ids) <= 5:
            # Use exhaustive search for small problems
            route, total_time = self.exhaustive_search(location_ids)
            metadata['algorithm'] = 'exhaustive_search'
            metadata['optimal'] = True
        else:
            # Use greedy + 2-opt for larger problems
            route, _ = self.greedy_nearest_neighbor(location_ids, start_location_id)
            route, total_time = self.two_opt_improvement(route)
            metadata['algorithm'] = 'greedy_plus_2opt'
            metadata['optimal'] = False
        
        metadata['total_drive_time_minutes'] = total_time
        
        return route, total_time, metadata
    
    def get_route_details(self, route: List[int]) -> List[Dict[str, any]]:
        """
        Get detailed information about each step in the route.
        
        Args:
            route: List of location IDs in order
            
        Returns:
            List of step details including drive times
        """
        if len(route) <= 1:
            return []
        
        details = []
        cumulative_time = 0.0
        
        for i in range(len(route) - 1):
            from_id = route[i]
            to_id = route[i + 1]
            drive_time = self.get_drive_time(from_id, to_id)
            cumulative_time += drive_time
            
            details.append({
                'step': i + 1,
                'from_location_id': from_id,
                'to_location_id': to_id,
                'drive_time_minutes': drive_time,
                'cumulative_time_minutes': cumulative_time
            })
        
        return details


def load_location_names(locations_path: str = "data/subway_locations.json") -> Dict[int, str]:
    """Load location ID to name mapping."""
    import json
    
    with open(locations_path, 'r') as f:
        data = json.load(f)
    
    return {
        loc['id']: loc['name'] 
        for loc in data['subway_locations_san_francisco']
    }


if __name__ == "__main__":
    # Example usage
    optimizer = DailyRouteOptimizer()
    
    # Test with a sample set of locations
    test_locations = [2, 3, 4, 5, 6]  # Sample location IDs
    
    print("Route Optimization Example:")
    print("==========================")
    
    # Optimize route
    route, total_time, metadata = optimizer.optimize_route(test_locations)
    
    # Load location names for display
    location_names = load_location_names()
    
    print(f"Algorithm used: {metadata['algorithm']}")
    print(f"Optimal solution: {metadata['optimal']}")
    print(f"Total drive time: {total_time:.1f} minutes")
    print(f"\nOptimized route:")
    
    for i, location_id in enumerate(route):
        print(f"  {i + 1}. {location_names.get(location_id, f'Location {location_id}')}")
    
    # Show step-by-step details
    print(f"\nRoute details:")
    details = optimizer.get_route_details(route)
    for step in details:
        from_name = location_names.get(step['from_location_id'], f"Location {step['from_location_id']}")
        to_name = location_names.get(step['to_location_id'], f"Location {step['to_location_id']}")
        print(f"  Step {step['step']}: {from_name} → {to_name}")
        print(f"    Drive time: {step['drive_time_minutes']:.1f} min, Cumulative: {step['cumulative_time_minutes']:.1f} min")