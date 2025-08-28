"""
Main Route Optimization Orchestrator

Coordinates Stage 1 (day assignment) and Stage 2 (route optimization) 
to solve the complete route optimization problem.
"""

import json
import yaml
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .stage1_assignment import DayAssignmentOptimizer
from .stage2_routing import DailyRouteOptimizer


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
    """Complete optimization result."""
    primary_assignments: Dict[int, int]  # day -> primary_location_id
    secondary_assignments: Dict[int, List[int]]  # day -> list of secondary_location_ids  
    daily_routes: Dict[int, List[int]]  # day -> optimized route
    daily_drive_times: Dict[int, float]  # day -> total drive time
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
    """Main route optimization orchestrator."""
    
    def __init__(
        self,
        config_path: str = "config/model-params.yaml",
        locations_path: str = "data/subway_locations.json",
        od_matrix_path: str = "data/od_matrix.pkl"
    ):
        """
        Initialize route optimizer.
        
        Args:
            config_path: Path to configuration YAML file
            locations_path: Path to locations JSON file  
            od_matrix_path: Path to OD matrix pickle file
        """
        self.config = OptimizationConfig.from_yaml(config_path)
        self.locations = self._load_locations(locations_path)
        
        # Initialize stage optimizers
        self.day_assigner = DayAssignmentOptimizer(od_matrix_path)
        self.route_optimizer = DailyRouteOptimizer(od_matrix_path)
        
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
            # Stage 1: Assign secondary locations to available days
            secondary_clusters = self.day_assigner.assign_days(
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
            
            # Primary days (single location, no routing needed)
            for day, primary_id in primary_assignments.items():
                daily_routes[day] = [primary_id]
                daily_drive_times[day] = 0.0
            
            # Secondary days (optimize routes)
            for day, location_ids in secondary_assignments.items():
                route, drive_time, route_metadata = self.route_optimizer.optimize_route(
                    location_ids=location_ids,
                    use_exhaustive_if_small=True
                )
                daily_routes[day] = route
                daily_drive_times[day] = drive_time
        
        end_time = datetime.now()
        
        # Compile metadata
        metadata = {
            'optimization_start_time': start_time.isoformat(),
            'optimization_end_time': end_time.isoformat(),
            'optimization_duration_seconds': (end_time - start_time).total_seconds(),
            'config': self.config.__dict__,
            'n_primary_locations': len(self.primary_locations),
            'n_secondary_locations': len(self.secondary_locations),
            'available_secondary_days': available_secondary_days,
            'stage1_quality_metrics': self.day_assigner.calculate_cluster_quality(secondary_clusters) 
                                    if available_secondary_days > 0 else {}
        }
        
        return OptimizationResult(
            primary_assignments=primary_assignments,
            secondary_assignments=secondary_assignments,
            daily_routes=daily_routes,
            daily_drive_times=daily_drive_times,
            metadata=metadata
        )
    
    def print_solution(self, result: OptimizationResult) -> None:
        """Print human-readable optimization results."""
        print("Route Optimization Results")
        print("=" * 50)
        
        # Summary statistics
        print(f"Total locations: {result.total_locations_visited()}")
        print(f"Total drive time: {result.total_drive_time():.1f} minutes")
        print(f"Optimization time: {result.metadata['optimization_duration_seconds']:.2f} seconds")
        print()
        
        # Create location lookup
        location_lookup = {loc.id: loc.name for loc in self.locations}
        
        # Print daily schedules
        all_days = sorted(set(result.primary_assignments.keys()) | set(result.secondary_assignments.keys()))
        
        for day in all_days:
            print(f"Day {day}:")
            
            if day in result.primary_assignments:
                primary_id = result.primary_assignments[day]
                print(f"  PRIMARY: {location_lookup[primary_id]} (full day)")
                print(f"  Drive time: 0.0 minutes")
            
            elif day in result.secondary_assignments:
                location_ids = result.secondary_assignments[day]
                route = result.daily_routes[day]
                drive_time = result.daily_drive_times[day]
                
                print(f"  SECONDARY ({len(location_ids)} locations):")
                print(f"  Route: {' → '.join([location_lookup[loc_id] for loc_id in route])}")
                print(f"  Drive time: {drive_time:.1f} minutes")
                
                # Show route details if requested
                details = self.route_optimizer.get_route_details(route)
                for step in details:
                    from_name = location_lookup[step['from_location_id']]
                    to_name = location_lookup[step['to_location_id']]
                    print(f"    {from_name} → {to_name}: {step['drive_time_minutes']:.1f} min")
            
            print()
        
        # Quality metrics
        if result.metadata['stage1_quality_metrics']:
            print("Clustering Quality Metrics:")
            for metric, value in result.metadata['stage1_quality_metrics'].items():
                print(f"  {metric}: {value:.2f}")
    
    def save_solution(self, result: OptimizationResult, output_path: str) -> None:
        """Save optimization results to JSON file."""
        # Convert result to serializable format
        solution_data = {
            'primary_assignments': result.primary_assignments,
            'secondary_assignments': result.secondary_assignments,
            'daily_routes': result.daily_routes,
            'daily_drive_times': result.daily_drive_times,
            'metadata': result.metadata,
            'summary': {
                'total_locations_visited': result.total_locations_visited(),
                'total_drive_time_minutes': result.total_drive_time()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(solution_data, f, indent=2, default=str)
        
        print(f"Solution saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    optimizer = RouteOptimizer()
    
    print("Starting route optimization...")
    result = optimizer.optimize()
    
    # Display results
    optimizer.print_solution(result)
    
    # Save results
    optimizer.save_solution(result, "data/optimization_result.json")