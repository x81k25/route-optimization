"""
Tests for Stage 1: Day Assignment Algorithm
"""

import pytest
import pickle
import numpy as np
import polars as pl
import tempfile
import json
from pathlib import Path

from src.core.stage1_assignment import DayAssignmentOptimizer, load_locations_data


class TestDayAssignmentOptimizer:
    
    @pytest.fixture
    def sample_od_matrix(self):
        """Create sample OD matrix for testing."""
        # Create small test dataset with 5 locations
        location_ids = [1, 2, 3, 4, 5]
        
        data = []
        for origin in location_ids:
            for dest in location_ids:
                if origin == dest:
                    drive_time = 0.0
                else:
                    # Create somewhat realistic drive times
                    drive_time = abs(origin - dest) * 2 + np.random.uniform(1, 3)
                
                data.append({
                    'origin_id': origin,
                    'destination_id': dest,
                    'origin_name': f'Location {origin}',
                    'destination_name': f'Location {dest}',
                    'distance_km': drive_time / 2,  # Rough approximation
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
        """Create DayAssignmentOptimizer instance."""
        return DayAssignmentOptimizer(od_matrix_file)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.n_locations == 5
        assert len(optimizer.location_ids) == 5
        assert optimizer.distance_matrix.shape == (5, 5)
    
    def test_distance_matrix_symmetry(self, optimizer):
        """Test that distance matrix is symmetric."""
        dist_matrix = optimizer.distance_matrix
        assert np.allclose(dist_matrix, dist_matrix.T)
    
    def test_distance_matrix_diagonal_zeros(self, optimizer):
        """Test that diagonal elements are zero."""
        dist_matrix = optimizer.distance_matrix
        assert np.allclose(np.diag(dist_matrix), 0)
    
    def test_cluster_locations_basic(self, optimizer):
        """Test basic clustering functionality."""
        clusters = optimizer.cluster_locations(n_clusters=2, max_locations_per_cluster=7)
        
        assert len(clusters) == 2  # Should have 2 clusters
        
        # All locations should be assigned
        all_assigned = set()
        for locations in clusters.values():
            all_assigned.update(locations)
        assert all_assigned == set(optimizer.location_ids)
        
        # No cluster should exceed size limit
        for locations in clusters.values():
            assert len(locations) <= 7
    
    def test_cluster_size_constraint_enforcement(self, optimizer):
        """Test that cluster size constraints are properly enforced."""
        # Force a small max size to trigger splitting
        clusters = optimizer.cluster_locations(n_clusters=1, max_locations_per_cluster=3)
        
        # Should create multiple clusters due to size constraint
        assert len(clusters) >= 2
        
        # Each cluster should respect size limit
        for locations in clusters.values():
            assert len(locations) <= 3
    
    def test_cluster_quality_calculation(self, optimizer):
        """Test cluster quality metrics calculation."""
        clusters = {1: [1, 2], 2: [3, 4, 5]}
        quality = optimizer.calculate_cluster_quality(clusters)
        
        assert 'avg_intra_cluster_distance' in quality
        assert 'cluster_size_std' in quality
        assert 'min_cluster_size' in quality
        assert 'max_cluster_size' in quality
        assert 'n_clusters' in quality
        
        assert quality['n_clusters'] == 2
        assert quality['min_cluster_size'] == 2
        assert quality['max_cluster_size'] == 3
    
    def test_swap_optimization(self, optimizer):
        """Test location swap optimization."""
        initial_clusters = {1: [1, 2], 2: [3, 4, 5]}
        improved_clusters = optimizer.optimize_with_swaps(initial_clusters, max_iterations=10)
        
        # Should return valid clusters
        assert len(improved_clusters) == 2
        
        # All locations should still be assigned
        all_assigned = set()
        for locations in improved_clusters.values():
            all_assigned.update(locations)
        assert all_assigned == {1, 2, 3, 4, 5}
    
    def test_assign_days_integration(self, optimizer):
        """Test complete day assignment workflow."""
        assignments = optimizer.assign_days(
            available_secondary_days=2,
            max_locations_per_day=7,
            use_swap_optimization=True
        )
        
        assert len(assignments) == 2  # Should have 2 days
        
        # All locations should be assigned
        all_assigned = set()
        for locations in assignments.values():
            all_assigned.update(locations)
        assert all_assigned == set(optimizer.location_ids)
        
        # Each day should respect location limit
        for locations in assignments.values():
            assert len(locations) <= 7
    
    def test_empty_clustering(self):
        """Test handling of edge cases."""
        # Create empty OD matrix
        empty_od = pl.DataFrame({
            'origin_id': [],
            'destination_id': [],
            'origin_name': [],
            'destination_name': [],
            'distance_km': [],
            'drive_time_minutes': []
        })
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pickle.dump(empty_od, tmp)
            tmp.flush()
            
            optimizer = DayAssignmentOptimizer(tmp.name)
            assert optimizer.n_locations == 0
            
            assignments = optimizer.assign_days(available_secondary_days=2)
            assert len(assignments) == 0


class TestLocationDataLoading:
    
    @pytest.fixture
    def sample_locations_file(self, tmp_path):
        """Create sample locations JSON file."""
        locations_data = {
            "subway_locations_san_francisco": [
                {
                    "id": 1,
                    "name": "Primary Location",
                    "class": "primary",
                    "address": "123 Main St",
                    "latitude": 37.7749,
                    "longitude": -122.4194
                },
                {
                    "id": 2,
                    "name": "Secondary Location 1",
                    "class": "secondary", 
                    "address": "456 Oak St",
                    "latitude": 37.7849,
                    "longitude": -122.4094
                },
                {
                    "id": 3,
                    "name": "Secondary Location 2",
                    "class": "secondary",
                    "address": "789 Pine St", 
                    "latitude": 37.7949,
                    "longitude": -122.3994
                }
            ]
        }
        
        locations_file = tmp_path / "test_locations.json"
        with open(locations_file, 'w') as f:
            json.dump(locations_data, f)
        
        return str(locations_file)
    
    def test_load_locations_data(self, sample_locations_file):
        """Test loading location data from JSON."""
        locations = load_locations_data(sample_locations_file)
        
        # Should only return secondary locations
        assert len(locations) == 2
        for loc in locations:
            assert loc['class'] == 'secondary'
            assert 'id' in loc
            assert 'name' in loc
            assert 'address' in loc
            assert 'latitude' in loc
            assert 'longitude' in loc


if __name__ == "__main__":
    pytest.main([__file__])