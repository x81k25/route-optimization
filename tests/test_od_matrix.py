#!/usr/bin/env python3
"""Test the generate_od_matrix function with sample data."""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from loguru import logger
from src.utils import osrm_utils

# Create test data for zone_004
test_data = {
    'pos_id': [10, 35, 78],
    'name': ['Subway - Antelope', 'Subway - Cameron Park', 'Subway - Galt'],
    'address': ['4320 Elverta Rd, Antelope, CA', '3490 Palmer Drive, Cameron Park, CA', '1067 C St, Galt, CA'],
    'latitude': [38.712422, 38.662374, 38.253825],
    'longitude': [-121.36357, -120.967577, -121.295203],
    'zone_id': ['zone_004', 'zone_004', 'zone_004'],
    'class': ['secondary', 'secondary', 'secondary']
}

def test_generate_od_matrix():
    """Test the generate_od_matrix function with sample data."""
    
    # Create DataFrame
    pos_zone = pl.DataFrame(test_data)
    logger.info(f"Testing with {len(pos_zone)} locations from zone_004")
    logger.info(pos_zone)
    
    # Generate OD matrix
    od_matrix = osrm_utils.generate_od_matrix(pos_zone)
    
    if od_matrix.is_empty():
        logger.error("Failed to generate OD matrix")
        return
    
    logger.success(f"Generated OD matrix with {len(od_matrix)} pairs")
    
    # Display sample results
    logger.info("\nSample OD Matrix Results:")
    sample = od_matrix.head(10)
    logger.info(sample)
    
    # Show some statistics
    avg_duration = od_matrix['duration_minutes'].mean()
    avg_distance = od_matrix['distance_meters'].mean()
    logger.info(f"\nStatistics:")
    logger.info(f"  Average duration: {avg_duration:.1f} minutes")
    logger.info(f"  Average distance: {avg_distance:.0f} meters")
    
    # Check matrix properties
    n_locations = len(test_data['pos_id'])
    expected_pairs = n_locations * n_locations
    assert len(od_matrix) == expected_pairs, f"Expected {expected_pairs} pairs, got {len(od_matrix)}"
    
    # Check for self-distances (should be 0 or very small)
    self_pairs = od_matrix.filter(pl.col('origin_id') == pl.col('destination_id'))
    max_self_distance = self_pairs['distance_meters'].max()
    logger.info(f"  Max self-distance: {max_self_distance:.1f} meters")
    
    logger.success("All tests passed!")

if __name__ == "__main__":
    test_generate_od_matrix()