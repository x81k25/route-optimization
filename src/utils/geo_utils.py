#!/usr/bin/env python3
"""
Geocoding utility script to fix coordinates in subway_locations.jsonl

This script:
1. Loads the subway_locations.jsonl file
2. Geocodes each address to get correct lat/lon coordinates
3. Updates the coordinates in the data
4. Rewrites the file with corrected coordinates

Usage:
    python src/utils/geocoding_utils.py
    
Or from project root:
    uv run python src/utils/geocoding_utils.py
"""

# standard library imports
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
import requests

# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Location:
    """
    Data class for location information.
    
    :param id: unique location identifier
    :param name: location name
    :param address: street address
    :param latitude: latitude coordinate
    :param longitude: longitude coordinate
    :param zone_id: zone identifier
    :param class_type: location classification
    """
    id: int
    name: str
    address: str
    latitude: float
    longitude: float
    zone_id: str
    class_type: str

class GeocodeProvider:
    """Base class for geocoding providers"""
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address and return (lat, lon) or None if failed"""
        raise NotImplementedError

class NominatimGeocoder(GeocodeProvider):
    """OpenStreetMap Nominatim geocoder"""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RouteOptimization/1.0 (research project)'
        })
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode address using Nominatim"""
        try:
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'countrycodes': 'us',  # restrict to US since all addresses are California
                'addressdetails': 1
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            if results:
                result = results[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                logger.debug(f"geocoded '{address}' -> {lat:.4f}, {lon:.4f}")
                return (lat, lon)
            else:
                logger.warning(f"No results for address: {address}")
                return None
                
        except Exception as e:
            logger.error(f"Geocoding failed for '{address}': {e}")
            return None

class GoogleGeocoder(GeocodeProvider):
    """Google Geocoding API (requires API key)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.session = requests.Session()
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode address using Google Geocoding API"""
        try:
            params = {
                'address': address,
                'key': self.api_key,
                'region': 'us'  # bias towards US results
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                lat = float(location['lat'])
                lon = float(location['lng'])
                logger.debug(f"geocoded '{address}' -> {lat:.4f}, {lon:.4f}")
                return (lat, lon)
            else:
                logger.warning(f"Google geocoding failed for '{address}': {data['status']}")
                return None
                
        except Exception as e:
            logger.error(f"Google geocoding failed for '{address}': {e}")
            return None

def load_locations(
    file_path: Path
) -> List[Location]:
    """
    Load locations from JSONL file.
    
    :param file_path: path to JSONL file
    :return: list of Location objects
    """
    locations = []
    
    logger.info(f"loading locations from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                location = Location(
                    id=data['id'],
                    name=data['name'],
                    address=data['address'],
                    latitude=float(data['latitude']),
                    longitude=float(data['longitude']),
                    zone_id=data['zone_id'],
                    class_type=data['class']
                )
                locations.append(location)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"loaded {len(locations)} locations")
    return locations

def save_locations(
    locations: List[Location], 
    file_path: Path
) -> None:
    """
    Save locations to JSONL file.
    
    :param locations: list of Location objects
    :param file_path: path where to save JSONL file
    :return: None
    """
    logger.info(f"saving {len(locations)} locations to {file_path}")
    
    # create backup of original file
    backup_path = file_path.with_suffix('.jsonl.backup')
    if file_path.exists():
        logger.info(f"creating backup at {backup_path}")
        backup_path.write_bytes(file_path.read_bytes())
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for location in locations:
            data = {
                'id': location.id,
                'name': location.name,
                'address': location.address,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'zone_id': location.zone_id,
                'class': location.class_type
            }
            f.write(json.dumps(data) + '\n')
    
    logger.info(f"successfully saved locations to {file_path}")

def geocode_single_location(location: Location, geocoder: GeocodeProvider, index: int, total: int) -> Tuple[Location, bool]:
    """Geocode a single location and return updated location with success flag"""
    logger.info(f"Geocoding {index}/{total}: {location.name}")
    logger.info(f"  Address: {location.address}")
    logger.info(f"  Current: {location.latitude:.6f}, {location.longitude:.6f}")
    
    # geocode the address
    result = geocoder.geocode(location.address)
    
    if result:
        new_lat, new_lon = result
        old_lat, old_lon = location.latitude, location.longitude
        
        # calculate distance moved (rough approximation in degrees)
        distance = ((new_lat - old_lat)**2 + (new_lon - old_lon)**2)**0.5
        
        logger.info(f"  Updated: {new_lat:.6f}, {new_lon:.6f}")
        logger.info(f"  Moved: {distance:.4f} degrees (~{distance*69:.1f} miles)")
        
        # update the location
        location.latitude = new_lat
        location.longitude = new_lon
        logger.info("")  # empty line for readability
        return location, True
    else:
        logger.warning(f"  Failed to geocode: {location.address} - setting coordinates to null")
        # set coordinates to None for failed geocoding
        location.latitude = None
        location.longitude = None
        logger.info("")  # empty line for readability
        return location, False


def geocode_locations(locations: List[Location], geocoder: GeocodeProvider, max_workers: int = 8) -> int:
    """Geocode all locations concurrently and update their coordinates"""
    
    updated_count = 0
    failed_count = 0
    
    logger.info(f"Starting concurrent geocoding of {len(locations)} locations with {max_workers} workers")
    
    # create thread-local geocoder instances to avoid session conflicts
    thread_local = threading.local()
    
    def get_thread_geocoder():
        if not hasattr(thread_local, 'geocoder'):
            thread_local.geocoder = NominatimGeocoder()
        return thread_local.geocoder
    
    def geocode_worker(location_with_index):
        location, index = location_with_index
        thread_geocoder = get_thread_geocoder()
        return geocode_single_location(location, thread_geocoder, index + 1, len(locations))
    
    # prepare locations with their indices
    locations_with_indices = [(location, i) for i, location in enumerate(locations)]
    
    # process locations concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_location = {executor.submit(geocode_worker, loc_idx): loc_idx[0] 
                             for loc_idx in locations_with_indices}
        
        # process completed tasks
        for future in as_completed(future_to_location):
            try:
                updated_location, success = future.result()
                if success:
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error geocoding location: {e}")
                failed_count += 1
    
    logger.info(f"Concurrent geocoding completed: {updated_count} updated, {failed_count} failed (set to null)")
    return updated_count


# ------------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------------

def get_centroid(pos_zone):
    """
    Calculate the centroid (geometric center) of coordinates in a dataframe.
    
    :param pos_zone: Polars DataFrame containing latitude and longitude columns
    :return: tuple (latitude, longitude) representing the centroid
    """
    centroid_lat = pos_zone['latitude'].mean()
    centroid_lon = pos_zone['longitude'].mean()
    return (centroid_lat, centroid_lon)


def main():
    """
    Main function to run geocoding process.
    
    :return: exit code (0 for success, 1 for failure)
    """
    # file paths
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "subway_locations.jsonl"
    
    if not data_file.exists():
        logger.error(f"data file not found: {data_file}")
        return 1
    
    # load locations
    try:
        locations = load_locations(data_file)
    except Exception as e:
        logger.error(f"failed to load locations: {e}")
        return 1
    
    if not locations:
        logger.error("no locations loaded")
        return 1
    
    # initialize geocoder (using Nominatim as free option)
    logger.info("using nominatim geocoder (openstreetmap)")
    logger.info("no rate limiting - concurrent processing with 8 workers")
    geocoder = NominatimGeocoder()
    
    # for Google Geocoder (if you have an API key):
    # google_api_key = "YOUR_GOOGLE_API_KEY_HERE"
    # geocoder = GoogleGeocoder(google_api_key)
    
    # geocode all locations concurrently
    try:
        updated_count = geocode_locations(locations, geocoder, max_workers=8)
    except KeyboardInterrupt:
        logger.info("geocoding interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"geocoding failed: {e}")
        return 1
    
    # always save results (including null coordinates for failed geocoding)
    try:
        save_locations(locations, data_file)
        logger.info(f"Successfully processed {len(locations)} locations ({updated_count} updated, {len(locations)-updated_count} failed)")
        
        # generate visualization map of all geocoded locations (including nulls)
        logger.info("Generating geocoding results visualization...")
        try:
            from .visualization import create_geocoding_results_map
            
            # convert Location objects to dictionaries for visualization
            # filter out locations with null coordinates for visualization
            locations_data = []
            for location in locations:
                if location.latitude is not None and location.longitude is not None:
                    locations_data.append({
                        'id': location.id,
                        'name': location.name,
                        'address': location.address,
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'zone_id': location.zone_id,
                        'class': location.class_type
                    })
            
            if locations_data:
                # generate the map
                output_dir = project_root / "output"
                map_path = create_geocoding_results_map(
                    locations_data=locations_data,
                    output_path=str(output_dir / "geocoding_results_map.html")
                )
                
                if map_path:
                    logger.info(f"Geocoding results map saved to: {map_path}")
                    logger.info(f"Map contains {len(locations_data)} successfully geocoded locations")
                else:
                    logger.warning("Failed to generate geocoding results map")
            else:
                logger.warning("No valid coordinates available for map generation")
                
        except ImportError as e:
            logger.warning(f"Could not generate visualization map: {e}")
        except Exception as e:
            logger.error(f"Failed to generate visualization map: {e}")
            
    except Exception as e:
        logger.error(f"Failed to save locations: {e}")
        return 1
    
    # final summary
    failed_count = len(locations) - updated_count
    logger.info("=" * 60)
    logger.info("GEOCODING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total locations processed: {len(locations)}")
    logger.info(f"Successfully geocoded: {updated_count}")
    logger.info(f"Failed (set to null): {failed_count}")
    logger.info(f"Success rate: {(updated_count/len(locations)*100):.1f}%")
    logger.info(f"Data saved to: {data_file}")
    if failed_count > 0:
        logger.info(f"NOTE: {failed_count} locations have null coordinates and will be excluded from route optimization")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())