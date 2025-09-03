#!/usr/bin/env python3
"""
Test timeout handling and error logging for OSRM API calls in isolation.
"""

import requests
import time
from src.utils.osrm_utils import fetch_od_matrix, validate_california_coordinates, Location
from src.core.stage0_data_ingestion import load_locations_from_jsonl
from loguru import logger

def test_timeout_handling():
    """Test timeout handling with problematic zones."""
    
    print("Testing timeout handling and error logging...")
    print("=" * 60)
    
    # Load zone_023 data
    try:
        df = load_locations_from_jsonl('data/subway_locations.jsonl', 'zone_023')
        print(f"✓ Loaded zone_023 with {len(df)} locations")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Convert to Location objects
    from src.utils.osrm_utils import convert_locations_from_polars
    locations = convert_locations_from_polars(df)
    
    print(f"✓ Converted to {len(locations)} Location objects")
    
    # Test coordinate validation
    valid_locs = validate_california_coordinates(locations, 'zone_023')
    print(f"✓ Validated coordinates: {len(valid_locs)} valid locations")
    
    # Test normal API call (should work)
    print("\n1. Testing normal API call:")
    try:
        result = fetch_od_matrix('zone_023', valid_locs[:2])  # Just 2 locations
        print(f"✓ Normal API call succeeded")
        print(f"  Response code: {result.osrm_response_code}")
        print(f"  Matrix shape: {result.duration_matrix.shape}")
    except Exception as e:
        print(f"✗ Normal API call failed: {e}")
    
    # Test with all locations (might timeout)
    print(f"\n2. Testing with all {len(valid_locs)} locations:")
    try:
        start_time = time.time()
        result = fetch_od_matrix('zone_023', valid_locs)
        end_time = time.time()
        
        print(f"✓ All locations API call succeeded in {end_time - start_time:.2f}s")
        print(f"  Response code: {result.osrm_response_code}")
        print(f"  Matrix shape: {result.duration_matrix.shape}")
        
        if result.osrm_response_code == "Fallback":
            print("  ⚠ Used fallback haversine distances")
        else:
            print("  ✓ Used OSRM API data")
            
    except Exception as e:
        print(f"✗ All locations API call failed: {e}")
    
    # Test with impossible coordinates (should use fallback)
    print(f"\n3. Testing with impossible coordinates:")
    impossible_locs = [
        Location(999, 0.0, 0.0, "Impossible Location 1"),
        Location(998, 0.0, 0.0, "Impossible Location 2")
    ]
    
    try:
        start_time = time.time()
        result = fetch_od_matrix('test_impossible', impossible_locs)
        end_time = time.time()
        
        print(f"✓ Impossible coordinates handled in {end_time - start_time:.2f}s")
        print(f"  Response code: {result.osrm_response_code}")
        print(f"  Matrix shape: {result.duration_matrix.shape}")
        
        if result.osrm_response_code == "Fallback":
            print("  ✓ Correctly used fallback for impossible coordinates")
        else:
            print("  ⚠ Unexpectedly got OSRM data")
            
    except Exception as e:
        print(f"✗ Impossible coordinates test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Timeout handling test completed!")

if __name__ == "__main__":
    test_timeout_handling()