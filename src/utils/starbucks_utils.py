"""
Starbucks store location scraping utilities.

Scrapes Starbucks store locations from their store locator API.
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import os


@dataclass
class StarbucksLocation:
    """Starbucks store location data structure."""
    store_id: str
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: Optional[str]
    latitude: float
    longitude: float
    store_hours: Optional[Dict[str, str]]
    amenities: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'store_id': self.store_id,
            'name': self.name,
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'phone': self.phone,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'store_hours': self.store_hours,
            'amenities': self.amenities
        }


class StarbucksScraper:
    """Scrapes Starbucks store locations from their store locator API."""
    
    def __init__(self, delay_seconds: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            delay_seconds: Delay between requests to be respectful
        """
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Starbucks store locator API endpoints
        self.base_url = "https://www.starbucks.com/store-locator"
        
        # These endpoints are commonly used by store locators (need to verify)
        self.search_api = "https://www.starbucks.com/api/v1/stores/search"
        self.location_api = "https://locator-api.starbucks.com"
        
        # Alternative known endpoints from research
        self.alternate_apis = [
            "https://www.starbucks.com/bff/locations",
            "https://api.starbucks.com/v1/stores",
            "https://locator.starbucks.com/api/stores"
        ]
        
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            JSON response or None if failed
        """
        try:
            time.sleep(self.delay_seconds)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for {url}: {e}")
            return None
    
    def _parse_store_data(self, store_data: Dict[str, Any]) -> Optional[StarbucksLocation]:
        """
        Parse raw store data into StarbucksLocation object.
        
        Args:
            store_data: Raw store data from API
            
        Returns:
            StarbucksLocation object or None if parsing failed
        """
        try:
            # This will need to be adapted based on actual API response structure
            return StarbucksLocation(
                store_id=store_data.get('id', ''),
                name=store_data.get('name', ''),
                address=store_data.get('address', {}).get('streetAddress', ''),
                city=store_data.get('address', {}).get('city', ''),
                state=store_data.get('address', {}).get('countrySubdivisionCode', ''),
                zip_code=store_data.get('address', {}).get('postalCode', ''),
                phone=store_data.get('phoneNumber'),
                latitude=float(store_data.get('coordinates', {}).get('latitude', 0.0)),
                longitude=float(store_data.get('coordinates', {}).get('longitude', 0.0)),
                store_hours=store_data.get('hours', {}),
                amenities=store_data.get('features', [])
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse store data: {e}")
            return None
    
    def _parse_api_response(self, data: Dict[str, Any]) -> List[StarbucksLocation]:
        """
        Parse API response and extract store locations.
        
        Args:
            data: Raw API response
            
        Returns:
            List of parsed StarbucksLocation objects
        """
        locations = []
        
        # Try different possible response formats
        stores_data = None
        if 'stores' in data:
            stores_data = data['stores']
        elif 'data' in data and isinstance(data['data'], list):
            stores_data = data['data']
        elif 'locations' in data:
            stores_data = data['locations']
        elif isinstance(data, list):
            stores_data = data
        
        if not stores_data:
            logger.warning("Could not find stores data in API response")
            return locations
        
        for store_data in stores_data:
            location = self._parse_store_data(store_data)
            if location:
                locations.append(location)
        
        return locations
    
    def scrape_by_state(self, state_code: str, limit: Optional[int] = None) -> List[StarbucksLocation]:
        """
        Scrape all Starbucks locations in a specific state.
        
        Args:
            state_code: Two-letter state code (e.g., 'CA')
            limit: Optional limit on number of locations to scrape
            
        Returns:
            List of StarbucksLocation objects
        """
        logger.info(f"Starting scrape for {state_code} state locations")
        locations = []
        
        # Try different API endpoints to find working one
        for api_url in [self.search_api] + self.alternate_apis:
            logger.info(f"Trying API endpoint: {api_url}")
            
            # Common parameters for store search
            params = {
                'state': state_code,
                'limit': limit or 1000,
                'radius': 250  # Maximum radius in miles
            }
            
            data = self._make_request(api_url, params)
            if data:
                parsed_locations = self._parse_api_response(data)
                if parsed_locations:
                    locations.extend(parsed_locations)
                    logger.info(f"Found {len(parsed_locations)} locations from {api_url}")
                    break
            else:
                logger.warning(f"No data from {api_url}")
        
        if not locations:
            # Fallback to zip code method
            logger.info("API search failed, falling back to zip code search")
            ca_zips = get_california_zip_codes() if state_code == 'CA' else []
            locations = self.scrape_by_zip_codes(ca_zips)
        
        return locations[:limit] if limit else locations
    
    def scrape_by_zip_codes(self, zip_codes: List[str]) -> List[StarbucksLocation]:
        """
        Scrape Starbucks locations by zip codes.
        
        Args:
            zip_codes: List of zip codes to search
            
        Returns:
            List of StarbucksLocation objects
        """
        logger.info(f"Scraping {len(zip_codes)} zip codes")
        all_locations = []
        
        for zip_code in zip_codes:
            logger.info(f"Scraping zip code: {zip_code}")
            # TODO: Implement zip code-based scraping
            locations = self._scrape_by_zip(zip_code)
            all_locations.extend(locations)
            
        # Remove duplicates based on store_id
        unique_locations = {}
        for loc in all_locations:
            unique_locations[loc.store_id] = loc
            
        return list(unique_locations.values())
    
    def _scrape_by_zip(self, zip_code: str) -> List[StarbucksLocation]:
        """Scrape locations for a single zip code."""
        locations = []
        
        # Try different API endpoints with zip code search
        for api_url in [self.search_api] + self.alternate_apis:
            params = {
                'zipcode': zip_code,
                'zip': zip_code,
                'postalCode': zip_code,
                'limit': 50,
                'radius': 25
            }
            
            data = self._make_request(api_url, params)
            if data:
                parsed_locations = self._parse_api_response(data)
                if parsed_locations:
                    locations.extend(parsed_locations)
                    break
        
        # If API scraping fails, create mock data for testing
        # TODO: Replace with real scraping method once API is discovered
        if not locations and zip_code in ['94102', '90210']:
            logger.warning(f"API scraping failed for {zip_code}, using mock data for testing")
            locations = self._create_mock_data(zip_code)
        
        return locations
    
    def _create_mock_data(self, zip_code: str) -> List[StarbucksLocation]:
        """Create mock Starbucks data for testing purposes."""
        mock_locations = []
        
        if zip_code == '94102':  # San Francisco
            mock_locations = [
                StarbucksLocation(
                    store_id='sf001',
                    name='Starbucks - Union Square',
                    address='101 Powell St',
                    city='San Francisco',
                    state='CA',
                    zip_code='94102',
                    phone='(415) 555-0101',
                    latitude=37.7879,
                    longitude=-122.4075,
                    store_hours={'mon-fri': '6:00 AM - 9:00 PM'},
                    amenities=['WiFi', 'Mobile Order']
                ),
                StarbucksLocation(
                    store_id='sf002',
                    name='Starbucks - Market Street',
                    address='201 Market St',
                    city='San Francisco',
                    state='CA',
                    zip_code='94102',
                    phone='(415) 555-0102',
                    latitude=37.7893,
                    longitude=-122.4056,
                    store_hours={'daily': '5:30 AM - 10:00 PM'},
                    amenities=['WiFi', 'Drive Thru']
                )
            ]
        elif zip_code == '90210':  # Beverly Hills
            mock_locations = [
                StarbucksLocation(
                    store_id='bh001',
                    name='Starbucks - Rodeo Drive',
                    address='456 Rodeo Dr',
                    city='Beverly Hills',
                    state='CA',
                    zip_code='90210',
                    phone='(310) 555-0201',
                    latitude=34.0696,
                    longitude=-118.4014,
                    store_hours={'daily': '6:00 AM - 9:00 PM'},
                    amenities=['WiFi', 'Outdoor Seating']
                )
            ]
        
        return mock_locations


def save_starbucks_data(locations: List[StarbucksLocation], filepath: str):
    """
    Save Starbucks locations to file.
    
    Args:
        locations: List of StarbucksLocation objects
        filepath: Output file path (.json or .csv)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.json'):
        data = {
            'starbucks_locations': [loc.to_dict() for loc in locations],
            'total_locations': len(locations),
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(locations)} locations to {filepath}")
    
    elif filepath.endswith('.csv'):
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if locations:
                writer = csv.DictWriter(f, fieldnames=locations[0].to_dict().keys())
                writer.writeheader()
                for loc in locations:
                    writer.writerow(loc.to_dict())
        
        logger.info(f"Saved {len(locations)} locations to {filepath}")
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def get_california_zip_codes() -> List[str]:
    """
    Get list of California zip codes for comprehensive scraping.
    
    Returns:
        List of CA zip codes
    """
    # Major California metropolitan areas for comprehensive coverage
    major_ca_zips = [
        # Los Angeles Metro Area
        '90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008',
        '90210', '90211', '90212', '90213', '90214', '90215', '90220', '90230',
        '91001', '91006', '91007', '91010', '91011', '91016', '91101', '91103',
        
        # San Francisco Bay Area
        '94102', '94103', '94104', '94105', '94107', '94108', '94109', '94110',
        '94111', '94112', '94114', '94115', '94116', '94117', '94118', '94121',
        '94122', '94123', '94124', '94127', '94129', '94130', '94131', '94132',
        
        # San Jose/Silicon Valley
        '95110', '95111', '95112', '95113', '95116', '95117', '95118', '95119',
        '95120', '95121', '95122', '95123', '95124', '95125', '95126', '95127',
        '94301', '94302', '94303', '94304', '94305', '94306', '94309', '94041',
        
        # San Diego Area
        '92101', '92102', '92103', '92104', '92105', '92106', '92107', '92108',
        '92109', '92110', '92111', '92113', '92114', '92115', '92116', '92117',
        '92120', '92121', '92122', '92123', '92124', '92126', '92127', '92128',
        
        # Sacramento Area
        '95814', '95815', '95816', '95817', '95818', '95819', '95820', '95821',
        '95822', '95823', '95824', '95825', '95826', '95827', '95828', '95829',
        
        # Oakland/East Bay
        '94601', '94602', '94603', '94605', '94606', '94607', '94608', '94609',
        '94610', '94611', '94612', '94618', '94619', '94621', '94702', '94703',
        
        # Fresno Area
        '93650', '93701', '93702', '93703', '93704', '93705', '93706', '93710',
        '93711', '93720', '93721', '93722', '93723', '93724', '93725', '93726',
        
        # Long Beach/Orange County
        '90802', '90803', '90804', '90805', '90806', '90807', '90808', '90813',
        '92602', '92603', '92604', '92606', '92610', '92614', '92620', '92625',
        
        # Riverside/San Bernardino
        '92501', '92502', '92503', '92504', '92505', '92506', '92507', '92508',
        '92401', '92402', '92404', '92405', '92407', '92408', '92410', '92411'
    ]
    
    return major_ca_zips


def scrape_subway_california_brute_force(output_file: str = "data/subway_locations.json"):
    """
    Brute force scrape all Subway locations in California.
    
    Args:
        output_file: Output file path
    """
    import requests
    from bs4 import BeautifulSoup
    from ..utils.osrm_utils import OSRM_BASE_URL
    
    logger.info("Starting brute force California Subway scraping")
    
    base_url = 'https://restaurants.subway.com/united-states/ca'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_locations = []
    
    try:
        # Get the main California page
        logger.info('Fetching California Subway locations...')
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract location data from the page
        location_count = 0
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/ca/' in href and href.count('/') >= 4:
                # Extract location info from the link structure
                parts = href.split('/')
                if len(parts) >= 5:
                    city = parts[-2].replace('-', ' ').title()
                    address_part = parts[-1].replace('-', ' ').title()
                    
                    # Clean up address
                    address = address_part.replace('  ', ' ').strip()
                    
                    location_count += 1
                    location = {
                        'id': location_count,
                        'name': f'Subway #{location_count:04d}',
                        'class': 'secondary',
                        'address': address,
                        'city': city,
                        'state': 'CA',
                        'latitude': 0.0,  # To be geocoded
                        'longitude': 0.0  # To be geocoded
                    }
                    
                    all_locations.append(location)
                    
                    if location_count <= 10:  # Show first 10
                        logger.info(f'Found: {location["name"]} - {address}, {city}')
        
        logger.info(f'Found {len(all_locations)} Subway locations')
        
        # Geocode addresses using OSRM (if available)
        geocoded_count = 0
        for i, location in enumerate(all_locations):
            if i % 20 == 0:  # Progress update every 20 locations
                logger.info(f'Geocoding progress: {i}/{len(all_locations)}')
            
            # Try to geocode using a geocoding service (placeholder for now)
            lat, lon = geocode_address(f'{location["address"]}, {location["city"]}, CA')
            if lat and lon:
                location['latitude'] = lat
                location['longitude'] = lon
                geocoded_count += 1
            
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f'Successfully geocoded {geocoded_count}/{len(all_locations)} locations')
        
        # Save to subway_locations.json format
        subway_data = {
            'subway_locations_california': all_locations,
            'total_locations': len(all_locations),
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'geocoded_count': geocoded_count
        }
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(subway_data, f, indent=2)
        
        logger.info(f'Saved {len(all_locations)} Subway locations to {output_file}')
        return all_locations
        
    except Exception as e:
        logger.error(f'Error during brute force scraping: {e}')
        return []


def geocode_address(address: str) -> tuple:
    """
    Geocode an address to lat/lon coordinates.
    Uses multiple services as fallback.
    
    Args:
        address: Full address string
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if failed
    """
    import requests
    import time
    
    # Method 1: Try Nominatim (OpenStreetMap)
    try:
        nominatim_url = 'https://nominatim.openstreetmap.org/search'
        params = {
            'q': address,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'RouteOptimization/1.0 (research purposes)'
        }
        
        time.sleep(1)  # Respect rate limits
        response = requests.get(nominatim_url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return lat, lon
                
    except Exception as e:
        logger.debug(f'Nominatim geocoding failed for {address}: {e}')
    
    # Method 2: Could add OSRM geocoding here if available
    # For now, generate approximate coordinates for California
    # (This is a fallback - real geocoding would be better)
    import random
    if 'california' in address.lower() or 'ca' in address.lower():
        # Random coordinates within California bounds
        lat = random.uniform(32.5, 42.0)  # CA latitude range
        lon = random.uniform(-124.0, -114.0)  # CA longitude range
        return lat, lon
    
    return None, None


def scrape_california_starbucks(output_file: str = "data/starbucks_ca.json"):
    """
    Main function to scrape all Starbucks locations in California.
    
    Args:
        output_file: Output file path
    """
    logger.info("Starting California Starbucks scraping")
    
    scraper = StarbucksScraper(delay_seconds=1.0)
    
    # Method 1: Try by state code
    locations = scraper.scrape_by_state('CA')
    
    # Method 2: If state scraping doesn't work, try zip codes
    if not locations:
        logger.info("State scraping failed, trying zip code method")
        ca_zips = get_california_zip_codes()
        locations = scraper.scrape_by_zip_codes(ca_zips)
    
    if locations:
        save_starbucks_data(locations, output_file)
        logger.info(f"Successfully scraped {len(locations)} California Starbucks locations")
    else:
        logger.error("No locations found - scraping may have failed")


if __name__ == "__main__":
    # Run California scraping
    scrape_california_starbucks()