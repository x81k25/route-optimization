#!/usr/bin/env python3
"""Test OSRM server connectivity and endpoints."""

import requests
from loguru import logger

# OSRM server configuration
OSRM_BASE_URL = "http://192.168.50.2:32050"

def test_osrm_server():
    """Test if OSRM server is accessible and responding."""
    
    # Test coordinates (San Francisco area)
    test_coords = "-122.4194,37.7749;-122.4094,37.7849"  # lon1,lat1;lon2,lat2
    
    # Test endpoints
    endpoints = {
        "table": f"{OSRM_BASE_URL}/table/v1/driving/{test_coords}?annotations=duration,distance",
        "route": f"{OSRM_BASE_URL}/route/v1/driving/{test_coords}?overview=false",
        "nearest": f"{OSRM_BASE_URL}/nearest/v1/driving/-122.4194,37.7749"
    }
    
    logger.info(f"Testing OSRM server at {OSRM_BASE_URL}")
    
    for endpoint_name, url in endpoints.items():
        try:
            logger.info(f"Testing {endpoint_name} endpoint...")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "Ok":
                    logger.success(f"✓ {endpoint_name} endpoint is working")
                    
                    # Show sample data
                    if endpoint_name == "table":
                        durations = data.get("durations", [])
                        if durations and len(durations) > 0:
                            logger.info(f"  Sample duration: {durations[0][1]:.1f} seconds")
                    elif endpoint_name == "route":
                        routes = data.get("routes", [])
                        if routes:
                            distance = routes[0].get("distance", 0)
                            duration = routes[0].get("duration", 0)
                            logger.info(f"  Route: {distance:.0f}m in {duration:.0f}s")
                    elif endpoint_name == "nearest":
                        waypoints = data.get("waypoints", [])
                        if waypoints:
                            distance = waypoints[0].get("distance", 0)
                            logger.info(f"  Nearest point: {distance:.1f}m away")
                else:
                    logger.warning(f"✗ {endpoint_name} returned: {data.get('code')} - {data.get('message', '')}")
            else:
                logger.error(f"✗ {endpoint_name} returned status code: {response.status_code}")
                
        except requests.ConnectionError:
            logger.error(f"✗ Cannot connect to {endpoint_name} endpoint - server may be down")
        except requests.Timeout:
            logger.error(f"✗ {endpoint_name} endpoint timed out")
        except Exception as e:
            logger.error(f"✗ {endpoint_name} endpoint error: {e}")
    
    logger.info("OSRM server test complete")

if __name__ == "__main__":
    test_osrm_server()