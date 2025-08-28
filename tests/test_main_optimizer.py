"""
Tests for Main Route Optimization Orchestrator
"""

import pytest
import json
import yaml
import tempfile
import pickle
import polars as pl
from pathlib import Path

from src.core.main_optimizer import RouteOptimizer, OptimizationConfig, OptimizationResult, Location


class TestOptimizationConfig:
    
    def test_config_creation(self):
        """Test direct configuration creation."""
        config = OptimizationConfig(
            days_per_week=5,
            utilization=100.0,
            primary_hours_per_week=24,
            hours_per_non_primary=1,
            locations_per_day_max=7,
            drive_inefficiency=0.0
        )
        
        assert config.days_per_week == 5
        assert config.utilization == 100.0
        assert config.primary_hours_per_week == 24
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {
            'model_params': {
                'days_per_week': 5,
                'utilization': 100,
                'primary_hours_per_week': 24,
                'hours_per_non_primary': 1,
                'locations_per_day_max': 7,
                'drive_inefficiency': 0.1,
                'start_location': 'primary'
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = OptimizationConfig.from_yaml(str(config_file))
        
        assert config.days_per_week == 5
        assert config.drive_inefficiency == 0.1
        assert config.start_location == 'primary'


class TestOptimizationResult:
    
    @pytest.fixture
    def sample_result(self):
        """Create sample optimization result."""
        return OptimizationResult(
            primary_assignments={1: 1},  # Day 1 -> Primary location 1
            secondary_assignments={
                2: [2, 3, 4],  # Day 2 -> Secondary locations 2, 3, 4
                3: [5, 6]      # Day 3 -> Secondary locations 5, 6
            },
            daily_routes={
                1: [1],
                2: [2, 3, 4],
                3: [5, 6]
            },
            daily_drive_times={
                1: 0.0,
                2: 25.5,
                3: 15.2
            },
            metadata={'test': True}
        )
    
    def test_total_drive_time(self, sample_result):
        """Test total drive time calculation."""
        total = sample_result.total_drive_time()
        assert total == 40.7  # 0.0 + 25.5 + 15.2
    
    def test_total_locations_visited(self, sample_result):
        """Test total locations count."""
        total = sample_result.total_locations_visited()
        assert total == 6  # 1 primary + 5 secondary


class TestRouteOptimizer:
    
    @pytest.fixture
    def test_files(self, tmp_path):
        """Create test configuration, locations, and OD matrix files."""
        # Configuration file
        config_data = {
            'model_params': {
                'days_per_week': 5,
                'utilization': 100,
                'primary_hours_per_week': 24,
                'hours_per_non_primary': 1,
                'locations_per_day_max': 7,
                'drive_inefficiency': 0.0,
                'start_location': 'primary'
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Locations file
        locations_data = {
            "subway_locations_san_francisco": [
                {
                    "id": 1,
                    "name": "Primary Location",
                    "class": "primary",
                    "address": "123 Primary St",
                    "latitude": 37.7749,
                    "longitude": -122.4194
                },
                {
                    "id": 2,
                    "name": "Secondary Location 1",
                    "class": "secondary",
                    "address": "456 Secondary Ave",
                    "latitude": 37.7849,
                    "longitude": -122.4094
                },
                {
                    "id": 3,
                    "name": "Secondary Location 2",
                    "class": "secondary",
                    "address": "789 Secondary Blvd",
                    "latitude": 37.7949,
                    "longitude": -122.3994
                },
                {
                    "id": 4,
                    "name": "Secondary Location 3",
                    "class": "secondary",
                    "address": "321 Secondary Way",
                    "latitude": 37.7649,
                    "longitude": -122.4294
                },
                {
                    "id": 5,
                    "name": "Secondary Location 4", 
                    "class": "secondary",
                    "address": "654 Secondary Rd",
                    "latitude": 37.7549,
                    "longitude": -122.4394
                }
            ]
        }
        locations_file = tmp_path / "locations.json"
        with open(locations_file, 'w') as f:
            json.dump(locations_data, f)
        
        # OD matrix (only secondary locations)
        secondary_ids = [2, 3, 4, 5]
        od_data = []
        for origin in secondary_ids:
            for dest in secondary_ids:
                if origin == dest:
                    drive_time = 0.0
                else:
                    drive_time = abs(origin - dest) * 5.0 + 3.0  # Simple distance model
                
                od_data.append({
                    'origin_id': origin,
                    'destination_id': dest,
                    'origin_name': f'Secondary Location {origin - 1}',
                    'destination_name': f'Secondary Location {dest - 1}',
                    'distance_km': drive_time / 2,
                    'drive_time_minutes': drive_time
                })
        
        od_matrix = pl.DataFrame(od_data)
        od_file = tmp_path / "od_matrix.pkl"
        with open(od_file, 'wb') as f:
            pickle.dump(od_matrix, f)
        
        return {
            'config_path': str(config_file),
            'locations_path': str(locations_file),
            'od_matrix_path': str(od_file)
        }
    
    @pytest.fixture
    def optimizer(self, test_files):
        """Create RouteOptimizer instance."""
        return RouteOptimizer(
            config_path=test_files['config_path'],
            locations_path=test_files['locations_path'],
            od_matrix_path=test_files['od_matrix_path']
        )
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.config.days_per_week == 5
        assert len(optimizer.locations) == 5
        assert len(optimizer.primary_locations) == 1
        assert len(optimizer.secondary_locations) == 4
        
        # Check location types
        primary = optimizer.primary_locations[0]
        assert primary.location_class == 'primary'
        assert primary.name == 'Primary Location'
        
        secondary = optimizer.secondary_locations[0]
        assert secondary.location_class == 'secondary'
    
    def test_assign_primary_days(self, optimizer):
        """Test primary location day assignment."""
        assignments = optimizer._assign_primary_days()
        
        assert len(assignments) == 1  # One primary location
        assert 1 in assignments
        assert assignments[1] == 1  # Primary location ID
    
    def test_calculate_available_secondary_days(self, optimizer):
        """Test available secondary days calculation."""
        available = optimizer._calculate_available_secondary_days()
        assert available == 4  # 5 total days - 1 primary day
    
    def test_optimize_integration(self, optimizer):
        """Test complete optimization workflow."""
        result = optimizer.optimize()
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert len(result.primary_assignments) == 1
        assert 1 in result.primary_assignments  # Day 1 has primary
        
        # Should have secondary assignments for remaining days
        assert len(result.secondary_assignments) > 0
        
        # All secondary locations should be assigned
        all_assigned_secondary = set()
        for locations in result.secondary_assignments.values():
            all_assigned_secondary.update(locations)
        
        secondary_ids = {loc.id for loc in optimizer.secondary_locations}
        assert all_assigned_secondary == secondary_ids
        
        # Check routes exist for all assigned days
        for day in result.primary_assignments:
            assert day in result.daily_routes
            assert day in result.daily_drive_times
        
        for day in result.secondary_assignments:
            assert day in result.daily_routes
            assert day in result.daily_drive_times
        
        # Check metadata
        assert 'optimization_start_time' in result.metadata
        assert 'optimization_end_time' in result.metadata
        assert 'n_primary_locations' in result.metadata
        assert 'n_secondary_locations' in result.metadata
        
        # Primary day should have zero drive time
        primary_day = list(result.primary_assignments.keys())[0]
        assert result.daily_drive_times[primary_day] == 0.0
    
    def test_no_secondary_days_available(self, test_files, tmp_path):
        """Test optimization when all days are consumed by primary locations."""
        # Create config with more primary locations than available days
        config_data = {
            'model_params': {
                'days_per_week': 2,  # Only 2 days available
                'utilization': 100,
                'primary_hours_per_week': 24,
                'hours_per_non_primary': 1,
                'locations_per_day_max': 7,
                'drive_inefficiency': 0.0
            }
        }
        config_file = tmp_path / "limited_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create locations with multiple primaries
        locations_data = {
            "subway_locations_san_francisco": [
                {"id": 1, "name": "Primary 1", "class": "primary", "address": "123 St", "latitude": 37.77, "longitude": -122.42},
                {"id": 2, "name": "Primary 2", "class": "primary", "address": "456 St", "latitude": 37.78, "longitude": -122.41},
                {"id": 3, "name": "Secondary 1", "class": "secondary", "address": "789 St", "latitude": 37.79, "longitude": -122.40}
            ]
        }
        locations_file = tmp_path / "limited_locations.json"
        with open(locations_file, 'w') as f:
            json.dump(locations_data, f)
        
        optimizer = RouteOptimizer(
            config_path=str(config_file),
            locations_path=str(locations_file),
            od_matrix_path=test_files['od_matrix_path']
        )
        
        result = optimizer.optimize()
        
        # Should have primary assignments but no secondary
        assert len(result.primary_assignments) == 2
        assert len(result.secondary_assignments) == 0
        
        # All routes should be single-location (primary only)
        for day, route in result.daily_routes.items():
            assert len(route) == 1
            assert result.daily_drive_times[day] == 0.0
    
    def test_save_solution(self, optimizer, tmp_path):
        """Test saving optimization results."""
        result = optimizer.optimize()
        output_file = tmp_path / "test_solution.json"
        
        optimizer.save_solution(result, str(output_file))
        
        # Verify file was created and contains expected data
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'primary_assignments' in saved_data
        assert 'secondary_assignments' in saved_data
        assert 'daily_routes' in saved_data
        assert 'daily_drive_times' in saved_data
        assert 'metadata' in saved_data
        assert 'summary' in saved_data
        
        # Check summary calculations
        summary = saved_data['summary']
        assert summary['total_locations_visited'] == result.total_locations_visited()
        assert abs(summary['total_drive_time_minutes'] - result.total_drive_time()) < 1e-6


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_locations(self, tmp_path):
        """Test handling of empty location sets."""
        # Empty locations file
        locations_data = {"subway_locations_san_francisco": []}
        locations_file = tmp_path / "empty_locations.json"
        with open(locations_file, 'w') as f:
            json.dump(locations_data, f)
        
        # Minimal config
        config_data = {'model_params': {'days_per_week': 5, 'utilization': 100, 'primary_hours_per_week': 24, 'hours_per_non_primary': 1, 'locations_per_day_max': 7, 'drive_inefficiency': 0.0}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Empty OD matrix
        od_matrix = pl.DataFrame({'origin_id': [], 'destination_id': [], 'origin_name': [], 'destination_name': [], 'distance_km': [], 'drive_time_minutes': []})
        od_file = tmp_path / "empty_od.pkl"
        with open(od_file, 'wb') as f:
            pickle.dump(od_matrix, f)
        
        optimizer = RouteOptimizer(
            config_path=str(config_file),
            locations_path=str(locations_file),
            od_matrix_path=str(od_file)
        )
        
        result = optimizer.optimize()
        
        assert len(result.primary_assignments) == 0
        assert len(result.secondary_assignments) == 0
        assert result.total_locations_visited() == 0
        assert result.total_drive_time() == 0.0


if __name__ == "__main__":
    pytest.main([__file__])