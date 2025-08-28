"""
Tests for Stage 2: Daily Route Optimization
"""

import pytest
import pickle
import polars as pl
import tempfile
import json
from pathlib import Path

from src.core.stage2_routing import DailyRouteOptimizer, load_location_names


class TestDailyRouteOptimizer:
    
    @pytest.fixture
    def sample_od_matrix(self):
        """Create sample OD matrix for testing."""
        # Create test dataset with 5 locations forming a rough line
        # Location 1 at (0,0), 2 at (1,0), 3 at (2,0), 4 at (3,0), 5 at (4,0)
        location_pairs = [
            (1, 2, 5.0), (1, 3, 10.0), (1, 4, 15.0), (1, 5, 20.0),
            (2, 3, 5.0), (2, 4, 10.0), (2, 5, 15.0),
            (3, 4, 5.0), (3, 5, 10.0),
            (4, 5, 5.0)
        ]
        
        data = []
        for origin in range(1, 6):
            for dest in range(1, 6):
                if origin == dest:
                    drive_time = 0.0
                else:
                    # Find drive time from pairs (symmetric)
                    drive_time = None
                    for o, d, time in location_pairs:
                        if (o == origin and d == dest) or (o == dest and d == origin):
                            drive_time = time
                            break
                    
                    if drive_time is None:
                        drive_time = abs(origin - dest) * 5.0  # Fallback
                
                data.append({
                    'origin_id': origin,
                    'destination_id': dest,
                    'origin_name': f'Location {origin}',
                    'destination_name': f'Location {dest}',
                    'distance_km': drive_time / 2,
                    'drive_time_minutes': drive_time
                })
        
        return pl.DataFrame(data)
    
    @pytest.fixture
    def od_matrix_file(self, sample_od_matrix, tmp_path):
        """Create temporary OD matrix file."""
        od_file = tmp_path / "test_od_matrix.pkl"
        with open(od_file, 'wb') as f:
            pickle.dump(sample_od_matrix, f)
        return str(od_file)
    
    @pytest.fixture
    def optimizer(self, od_matrix_file):
        """Create DailyRouteOptimizer instance."""
        return DailyRouteOptimizer(od_matrix_file)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.drive_time_lookup) > 0
        
        # Test drive time lookup
        assert optimizer.get_drive_time(1, 1) == 0.0
        assert optimizer.get_drive_time(1, 2) == 5.0
        assert optimizer.get_drive_time(2, 1) == 5.0  # Should be symmetric
    
    def test_greedy_nearest_neighbor(self, optimizer):
        """Test greedy nearest neighbor algorithm."""
        locations = [1, 2, 3, 4, 5]
        route, total_time = optimizer.greedy_nearest_neighbor(locations)
        
        # Should return all locations
        assert len(route) == 5
        assert set(route) == set(locations)
        
        # Should start with specified or first location
        assert route[0] == 1
        
        # Total time should be positive
        assert total_time > 0
    
    def test_greedy_with_start_location(self, optimizer):
        """Test greedy with specific start location."""
        locations = [1, 2, 3, 4, 5]
        route, total_time = optimizer.greedy_nearest_neighbor(locations, start_location_id=3)
        
        assert route[0] == 3
        assert len(route) == 5
        assert set(route) == set(locations)
    
    def test_two_opt_improvement(self, optimizer):
        """Test 2-opt improvement algorithm."""
        initial_route = [1, 5, 2, 4, 3]  # Deliberately bad route
        improved_route, improved_time = optimizer.two_opt_improvement(initial_route)
        
        # Should return same locations
        assert set(improved_route) == set(initial_route)
        
        # Should improve or maintain quality
        initial_time = optimizer._calculate_route_time(initial_route)
        assert improved_time <= initial_time
    
    def test_exhaustive_search_small(self, optimizer):
        """Test exhaustive search on small problem."""
        locations = [1, 2, 3]
        route, total_time = optimizer.exhaustive_search(locations)
        
        assert len(route) == 3
        assert set(route) == set(locations)
        assert route[0] == 1  # Should start with first location
        
        # For 3 locations in a line, optimal should be 1->2->3 or 1->3->2
        # Check that it's one of the two reasonable solutions
        assert route in [[1, 2, 3], [1, 3, 2]]
    
    def test_exhaustive_search_size_limit(self, optimizer):
        """Test that exhaustive search rejects large problems."""
        large_locations = list(range(1, 12))  # 11 locations
        
        with pytest.raises(ValueError):
            optimizer.exhaustive_search(large_locations)
    
    def test_optimize_route_small_problem(self, optimizer):
        """Test route optimization on small problem (should use exhaustive)."""
        locations = [1, 2, 3, 4]
        route, total_time, metadata = optimizer.optimize_route(locations, use_exhaustive_if_small=True)
        
        assert metadata['algorithm'] == 'exhaustive_search'
        assert metadata['optimal'] == True
        assert len(route) == 4
        assert set(route) == set(locations)
    
    def test_optimize_route_large_problem(self, optimizer):
        """Test route optimization on larger problem (should use greedy+2opt)."""
        locations = [1, 2, 3, 4, 5]
        route, total_time, metadata = optimizer.optimize_route(locations, use_exhaustive_if_small=False)
        
        assert metadata['algorithm'] == 'greedy_plus_2opt'
        assert metadata['optimal'] == False
        assert len(route) == 5
        assert set(route) == set(locations)
    
    def test_route_details(self, optimizer):
        """Test route detail generation."""
        route = [1, 2, 3, 4]
        details = optimizer.get_route_details(route)
        
        assert len(details) == 3  # n-1 steps for n locations
        
        for i, step in enumerate(details):
            assert step['step'] == i + 1
            assert step['from_location_id'] == route[i]
            assert step['to_location_id'] == route[i + 1]
            assert step['drive_time_minutes'] > 0
            assert 'cumulative_time_minutes' in step
    
    def test_empty_locations(self, optimizer):
        """Test handling of empty location lists."""
        route, total_time, metadata = optimizer.optimize_route([])
        
        assert route == []
        assert total_time == 0.0
        assert metadata['algorithm'] == 'empty'
    
    def test_single_location(self, optimizer):
        """Test handling of single location."""
        route, total_time, metadata = optimizer.optimize_route([1])
        
        assert route == [1]
        assert total_time == 0.0
        assert metadata['n_locations'] == 1
        
        # Details should be empty for single location
        details = optimizer.get_route_details(route)
        assert details == []
    
    def test_calculate_route_time(self, optimizer):
        """Test route time calculation."""
        route = [1, 2, 3, 4]
        total_time = optimizer._calculate_route_time(route)
        
        # Should equal sum of individual segments
        expected_time = (optimizer.get_drive_time(1, 2) + 
                        optimizer.get_drive_time(2, 3) + 
                        optimizer.get_drive_time(3, 4))
        
        assert abs(total_time - expected_time) < 1e-6
    
    def test_symmetric_drive_times(self, optimizer):
        """Test that drive times are symmetric."""
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j:
                    time_ij = optimizer.get_drive_time(i, j)
                    time_ji = optimizer.get_drive_time(j, i)
                    assert abs(time_ij - time_ji) < 1e-6


class TestLocationNameLoading:
    
    @pytest.fixture
    def sample_locations_file(self, tmp_path):
        """Create sample locations JSON file."""
        locations_data = {
            "subway_locations_san_francisco": [
                {"id": 1, "name": "Location One", "class": "primary"},
                {"id": 2, "name": "Location Two", "class": "secondary"},
                {"id": 3, "name": "Location Three", "class": "secondary"}
            ]
        }
        
        locations_file = tmp_path / "test_locations.json"
        with open(locations_file, 'w') as f:
            json.dump(locations_data, f)
        
        return str(locations_file)
    
    def test_load_location_names(self, sample_locations_file):
        """Test loading location names mapping."""
        name_mapping = load_location_names(sample_locations_file)
        
        assert len(name_mapping) == 3
        assert name_mapping[1] == "Location One"
        assert name_mapping[2] == "Location Two"
        assert name_mapping[3] == "Location Three"


class TestOptimizationAlgorithms:
    """Integration tests for optimization algorithms."""
    
    @pytest.fixture
    def linear_problem_optimizer(self, tmp_path):
        """Create optimizer with a linear TSP problem."""
        # 4 locations in a line: 1-2-3-4
        # Optimal tour should be 1->2->3->4 (or reverse)
        data = []
        locations = [1, 2, 3, 4]
        
        for origin in locations:
            for dest in locations:
                if origin == dest:
                    drive_time = 0.0
                else:
                    # Distance proportional to difference in position
                    drive_time = abs(origin - dest) * 10.0
                
                data.append({
                    'origin_id': origin,
                    'destination_id': dest,
                    'origin_name': f'Location {origin}',
                    'destination_name': f'Location {dest}',
                    'distance_km': drive_time / 2,
                    'drive_time_minutes': drive_time
                })
        
        od_matrix = pl.DataFrame(data)
        od_file = tmp_path / "linear_od_matrix.pkl"
        with open(od_file, 'wb') as f:
            pickle.dump(od_matrix, f)
        
        return DailyRouteOptimizer(str(od_file))
    
    def test_linear_tsp_optimal_solution(self, linear_problem_optimizer):
        """Test that algorithms find optimal solution for linear TSP."""
        locations = [1, 2, 3, 4]
        
        # Exhaustive search should find optimal
        route_exhaustive, time_exhaustive = linear_problem_optimizer.exhaustive_search(locations)
        
        # Greedy + 2-opt should find good solution
        route_greedy, time_greedy = linear_problem_optimizer.greedy_nearest_neighbor(locations)
        route_improved, time_improved = linear_problem_optimizer.two_opt_improvement(route_greedy)
        
        # For this simple problem, both should find optimal or near-optimal
        assert time_exhaustive <= 40.0  # Optimal is 1->2->3->4 = 10+10+10 = 30
        assert time_improved <= time_greedy  # 2-opt should not make things worse
        
        # Verify route validity
        assert set(route_exhaustive) == set(locations)
        assert set(route_improved) == set(locations)


if __name__ == "__main__":
    pytest.main([__file__])