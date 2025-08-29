"""
Main Route Optimization Orchestrator

Coordinates Stage 1 (day assignment) and Stage 2 (route optimization) 
to solve the complete route optimization problem.
"""

import json
import yaml
import polars as pl
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from .stage1_assignment import DayAssignmentOptimizer
from .stage2_routing import DailyRouteOptimizer
from ..utils.osrm_utils import fetch_route_geometry, convert_locations_from_polars, RouteGeometry


@dataclass
class OptimizationConfig:
    """Configuration parameters for route optimization."""
    days_per_week: int
    utilization: float
    primary_hours_per_week: int
    hours_per_non_primary: int
    locations_per_day_max: int
    drive_inefficiency: float
    start_location: str = "primary"
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/model-params.yaml"):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        params = data['model_params']
        return cls(**params)


@dataclass 
class Location:
    """Location data structure."""
    id: int
    name: str
    location_class: str  # 'primary' or 'secondary'
    address: str
    latitude: float
    longitude: float


@dataclass
class OptimizationResult:
    """Complete optimization result with route geometry."""
    primary_assignments: Dict[int, int]  # day -> primary_location_id
    secondary_assignments: Dict[int, List[int]]  # day -> list of secondary_location_ids  
    daily_routes: Dict[int, List[int]]  # day -> optimized route
    daily_drive_times: Dict[int, float]  # day -> total drive time
    route_geometries: Dict[int, RouteGeometry]  # day -> detailed route geometry
    metadata: Dict[str, Any]
    
    def total_drive_time(self) -> float:
        """Calculate total drive time across all days."""
        return sum(self.daily_drive_times.values())
    
    def total_locations_visited(self) -> int:
        """Calculate total number of locations visited."""
        total = len(self.primary_assignments)  # Primary locations
        total += sum(len(locs) for locs in self.secondary_assignments.values())  # Secondary locations
        return total


class RouteOptimizer:
    """Main route optimization orchestrator with OSRM integration."""
    
    def __init__(
        self,
        config_path: str = "config/model-params.yaml",
        locations_path: str = "data/subway_locations.json",
        zone_id: str = "default_zone"
    ):
        """
        Initialize route optimizer with OSRM integration.
        
        Args:
            config_path: Path to configuration YAML file
            locations_path: Path to locations JSON file  
            zone_id: Zone identifier for this optimization
        """
        self.config = OptimizationConfig.from_yaml(config_path)
        self.zone_id = zone_id
        self.locations = self._load_locations(locations_path)
        
        # Convert to Polars DataFrame for zone processing
        self.zone_df = self._locations_to_polars()
        
        # Separate primary and secondary locations
        self.primary_locations = [loc for loc in self.locations if loc.location_class == 'primary']
        self.secondary_locations = [loc for loc in self.locations if loc.location_class == 'secondary']
    
    def _load_locations(self, locations_path: str) -> List[Location]:
        """Load location data from JSON file."""
        with open(locations_path, 'r') as f:
            data = json.load(f)
        
        locations = []
        for loc_data in data['subway_locations_san_francisco']:
            locations.append(Location(
                id=loc_data['id'],
                name=loc_data['name'],
                location_class=loc_data['class'],
                address=loc_data['address'],
                latitude=loc_data['latitude'],
                longitude=loc_data['longitude']
            ))
        
        return locations
    
    def _locations_to_polars(self) -> pl.DataFrame:
        """Convert locations to Polars DataFrame."""
        data = []
        for loc in self.locations:
            data.append({
                'location_id': loc.id,
                'zone_id': self.zone_id,
                'name': loc.name,
                'location_class': loc.location_class,
                'address': loc.address,
                'latitude': loc.latitude,
                'longitude': loc.longitude,
                'source_system': 'json_file'
            })
        return pl.DataFrame(data)
    
    def _assign_primary_days(self) -> Dict[int, int]:
        """
        Assign primary locations to days.
        Simple assignment: one primary location per day.
        
        Returns:
            Dictionary mapping day -> primary_location_id
        """
        primary_assignments = {}
        
        for i, primary_location in enumerate(self.primary_locations):
            day = i + 1  # Days start from 1
            primary_assignments[day] = primary_location.id
        
        return primary_assignments
    
    def _calculate_available_secondary_days(self) -> int:
        """Calculate how many days are available for secondary locations."""
        primary_days_used = len(self.primary_locations)
        return max(0, self.config.days_per_week - primary_days_used)
    
    def optimize(self) -> OptimizationResult:
        """
        Main optimization method that coordinates both stages.
        
        Returns:
            Complete optimization result
        """
        start_time = datetime.now()
        
        # Stage 1: Assign primary locations to days
        primary_assignments = self._assign_primary_days()
        
        # Calculate available days for secondary locations
        available_secondary_days = self._calculate_available_secondary_days()
        
        if available_secondary_days <= 0:
            # No days available for secondary locations
            secondary_assignments = {}
            daily_routes = {day: [primary_id] for day, primary_id in primary_assignments.items()}
            daily_drive_times = {day: 0.0 for day in primary_assignments}
        else:
            # Initialize day assigner with zone data and get OSRM matrix
            secondary_df = self.zone_df.filter(pl.col('location_class') == 'secondary')
            day_assigner = DayAssignmentOptimizer(self.zone_id, secondary_df)
            
            # Stage 1: Assign secondary locations to available days
            secondary_clusters = day_assigner.assign_days(
                available_secondary_days=available_secondary_days,
                max_locations_per_day=self.config.locations_per_day_max,
                use_swap_optimization=True
            )
            
            # Convert cluster IDs to actual day numbers
            secondary_assignments = {}
            available_days = [d for d in range(1, self.config.days_per_week + 1) 
                            if d not in primary_assignments]
            
            for cluster_id, location_ids in secondary_clusters.items():
                if cluster_id - 1 < len(available_days):  # cluster_id starts from 1
                    day = available_days[cluster_id - 1]
                    secondary_assignments[day] = location_ids
            
            # Stage 2: Optimize routes for each day
            daily_routes = {}
            daily_drive_times = {}
            route_geometries = {}
            
            # Initialize route optimizer with OD matrix
            route_optimizer = DailyRouteOptimizer(day_assigner.get_od_matrix_polars())
            
            # Primary days (single location, no routing needed)
            for day, primary_id in primary_assignments.items():
                daily_routes[day] = [primary_id]
                daily_drive_times[day] = 0.0
                # No route geometry needed for single location
                route_geometries[day] = None
            
            # Secondary days (optimize routes)
            for day, location_ids in secondary_assignments.items():
                route, drive_time, route_metadata = route_optimizer.optimize_route(
                    location_ids=location_ids,
                    use_exhaustive_if_small=True
                )
                daily_routes[day] = route
                daily_drive_times[day] = drive_time
                
                # Fetch detailed route geometry
                if len(route) > 1:
                    route_locations = [loc for loc in day_assigner.locations if loc.location_id in route]
                    # Reorder locations according to optimized route
                    ordered_locations = []
                    for loc_id in route:
                        for loc in route_locations:
                            if loc.location_id == loc_id:
                                ordered_locations.append(loc)
                                break
                    
                    route_geometry = fetch_route_geometry(
                        zone_id=self.zone_id,
                        day_number=day,
                        route_locations=ordered_locations,
                        include_steps=True
                    )
                    route_geometries[day] = route_geometry
                else:
                    route_geometries[day] = None
        
        end_time = datetime.now()
        
        # Compile metadata
        metadata = {
            'optimization_start_time': start_time.isoformat(),
            'optimization_end_time': end_time.isoformat(),
            'optimization_duration_seconds': (end_time - start_time).total_seconds(),
            'config': self.config.__dict__,
            'zone_id': self.zone_id,
            'n_primary_locations': len(self.primary_locations),
            'n_secondary_locations': len(self.secondary_locations),
            'available_secondary_days': available_secondary_days,
            'stage1_quality_metrics': day_assigner.calculate_cluster_quality(secondary_clusters) 
                                    if available_secondary_days > 0 else {},
            'osrm_integration': True,
            'route_geometries_fetched': sum(1 for rg in route_geometries.values() if rg is not None)
        }
        
        return OptimizationResult(
            primary_assignments=primary_assignments,
            secondary_assignments=secondary_assignments,
            daily_routes=daily_routes,
            daily_drive_times=daily_drive_times,
            route_geometries=route_geometries,
            metadata=metadata
        )
    
    def print_solution(self, result: OptimizationResult) -> None:
        """Print human-readable optimization results."""
        logger.info("Route Optimization Results")
        logger.info("=" * 50)
        
        # Summary statistics
        logger.info(f"Total locations: {result.total_locations_visited()}")
        logger.info(f"Total drive time: {result.total_drive_time():.1f} minutes")
        logger.info(f"Optimization time: {result.metadata['optimization_duration_seconds']:.2f} seconds")
        logger.info(f"Zone ID: {result.metadata.get('zone_id', 'N/A')}")
        logger.info(f"Route geometries fetched: {result.metadata.get('route_geometries_fetched', 0)}")
        logger.info("")
        
        # Create location lookup
        location_lookup = {loc.id: loc.name for loc in self.locations}
        
        # Print daily schedules
        all_days = sorted(set(result.primary_assignments.keys()) | set(result.secondary_assignments.keys()))
        
        for day in all_days:
            logger.info(f"Day {day}:")
            
            if day in result.primary_assignments:
                primary_id = result.primary_assignments[day]
                logger.info(f"  PRIMARY: {location_lookup[primary_id]} (full day)")
                logger.info(f"  Drive time: 0.0 minutes")
            
            elif day in result.secondary_assignments:
                location_ids = result.secondary_assignments[day]
                route = result.daily_routes[day]
                drive_time = result.daily_drive_times[day]
                
                logger.info(f"  SECONDARY ({len(location_ids)} locations):")
                logger.info(f"  Route: {' → '.join([location_lookup[loc_id] for loc_id in route])}")
                logger.info(f"  Drive time: {drive_time:.1f} minutes")
                
                # Show route geometry info if available
                if day in result.route_geometries and result.route_geometries[day]:
                    route_geom = result.route_geometries[day]
                    logger.info(f"  Route geometry: {len(route_geom.turn_by_turn_instructions)} turn instructions")
                    logger.info(f"  Total distance: {route_geom.total_distance_meters:.0f} meters")
            
            logger.info("")
        
        # Quality metrics
        if result.metadata['stage1_quality_metrics']:
            logger.info("Clustering Quality Metrics:")
            for metric, value in result.metadata['stage1_quality_metrics'].items():
                logger.info(f"  {metric}: {value:.2f}")
    
    def save_solution(self, result: OptimizationResult, output_path: str) -> None:
        """Save optimization results to JSON file."""
        # Convert result to serializable format
        route_geometries_serializable = {}
        for day, route_geom in result.route_geometries.items():
            if route_geom:
                route_geometries_serializable[day] = {
                    'zone_id': route_geom.zone_id,
                    'day_number': route_geom.day_number,
                    'route_location_ids': route_geom.route_location_ids,
                    'geometry_polyline': route_geom.geometry_polyline,
                    'total_distance_meters': route_geom.total_distance_meters,
                    'total_duration_seconds': route_geom.total_duration_seconds,
                    'turn_by_turn_instructions': route_geom.turn_by_turn_instructions,
                    'osrm_response_code': route_geom.osrm_response_code,
                    'api_call_timestamp': route_geom.api_call_timestamp.isoformat()
                }
            else:
                route_geometries_serializable[day] = None
        
        solution_data = {
            'primary_assignments': result.primary_assignments,
            'secondary_assignments': result.secondary_assignments,
            'daily_routes': result.daily_routes,
            'daily_drive_times': result.daily_drive_times,
            'route_geometries': route_geometries_serializable,
            'metadata': result.metadata,
            'summary': {
                'total_locations_visited': result.total_locations_visited(),
                'total_drive_time_minutes': result.total_drive_time()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(solution_data, f, indent=2, default=str)
        
        logger.info(f"Solution saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    optimizer = RouteOptimizer(zone_id="sf_subway_zone")
    
    logger.info("Starting route optimization with OSRM integration...")
    result = optimizer.optimize()
    
    # Display results
    optimizer.print_solution(result)
    
    # Save results
    optimizer.save_solution(result, "output/optimization_result.json")