"""
Core Reporting and Output Pipeline

Handles the final stages from the architecture:
- Stage 5: Detailed Route Geometry Fetch (OSRM Route API per day)
- Stage 6: Route Metrics Summary Generation (Itinerary & analytics) 
- Zone Report Generator
- Visualization Generator
- JSON Results Export
"""

import json
import polars as pl
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger

from ..utils.visualization import create_route_map, save_route_visualization


def calculate_total_drive_time(daily_drive_times: Dict[int, float]) -> float:
    """Calculate total drive time across all days."""
    return sum(daily_drive_times.values())


def calculate_total_locations_visited(primary_assignments: Dict[int, int], 
                                    secondary_assignments: Dict[int, List[int]]) -> int:
    """Calculate total number of locations visited."""
    total = len(primary_assignments)  # Primary locations
    total += sum(len(locs) for locs in secondary_assignments.values())  # Secondary locations
    return total


def calculate_primary_store_count(primary_assignments: Dict[int, int]) -> int:
    """Calculate total number of primary locations."""
    return len(primary_assignments)


def calculate_non_primary_store_count(secondary_assignments: Dict[int, List[int]]) -> int:
    """Calculate total number of secondary/non-primary locations."""
    return sum(len(locs) for locs in secondary_assignments.values())


def calculate_total_primary_hours(config: Dict[str, Any]) -> float:
    """Calculate total hours spent at primary locations."""
    return config.get('primary_hours_per_week', 24)


def calculate_total_non_primary_hours(secondary_assignments: Dict[int, List[int]], 
                                    config: Dict[str, Any]) -> float:
    """Calculate total hours spent at non-primary locations."""
    hours_per_non_primary = config.get('hours_per_non_primary', 1)
    non_primary_count = calculate_non_primary_store_count(secondary_assignments)
    return non_primary_count * hours_per_non_primary


def calculate_unutilized_time(primary_assignments: Dict[int, int],
                            secondary_assignments: Dict[int, List[int]],
                            daily_drive_times: Dict[int, float],
                            config: Dict[str, Any]) -> float:
    """Calculate unutilized time in hours per week, including drive time only for secondary days."""
    days_per_week = config.get('days_per_week', 5)
    utilization = config.get('utilization', 100) / 100.0
    
    # Total available hours per week (assuming 8 hours per working day)
    total_available_hours = days_per_week * 8.0 * utilization
    
    # Used hours: location time + drive time (only for secondary days)
    location_hours = (calculate_total_primary_hours(config) + 
                     calculate_total_non_primary_hours(secondary_assignments, config))
    
    # Calculate drive time only for secondary days (not primary days)
    secondary_drive_time = 0.0
    for day, drive_time in daily_drive_times.items():
        # Only count drive time if this day has secondary assignments (not primary)
        if day in secondary_assignments:
            secondary_drive_time += drive_time
    
    drive_hours = secondary_drive_time / 60.0  # Convert minutes to hours
    used_hours = location_hours + drive_hours
    
    # Unutilized time (can be negative if over-utilized)
    return float(total_available_hours - used_hours)


def generate_route_metrics_summary(optimization_package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 6: Route Metrics Summary Generation - Generate comprehensive analytics.
    
    Args:
        optimization_package: Complete optimization results
        
    Returns:
        Dictionary containing detailed metrics and analytics
    """
    primary_assignments = optimization_package['primary_assignments']
    secondary_assignments = optimization_package['secondary_assignments']
    daily_drive_times = optimization_package['daily_drive_times']
    config = optimization_package['config']
    
    # Basic metrics
    total_locations = calculate_total_locations_visited(primary_assignments, secondary_assignments)
    total_drive_time = calculate_total_drive_time(daily_drive_times)
    primary_count = calculate_primary_store_count(primary_assignments)
    non_primary_count = calculate_non_primary_store_count(secondary_assignments)
    
    # Time utilization
    primary_hours = calculate_total_primary_hours(config)
    non_primary_hours = calculate_total_non_primary_hours(secondary_assignments, config)
    unutilized_time = calculate_unutilized_time(primary_assignments, secondary_assignments, daily_drive_times, config)
    
    # Route efficiency metrics
    avg_drive_time_per_day = total_drive_time / max(len(daily_drive_times), 1)
    avg_locations_per_secondary_day = (non_primary_count / max(len(secondary_assignments), 1)) if secondary_assignments else 0
    
    return {
        'total_locations_visited': total_locations,
        'primary_store_count': primary_count,
        'non_primary_store_count': non_primary_count,
        'total_drive_time_minutes': total_drive_time,
        'total_drive_time_hours': total_drive_time / 60.0,
        'total_primary_hours': primary_hours,
        'total_non_primary_hours': non_primary_hours,
        'unutilized_time_hours': unutilized_time,
        'avg_drive_time_per_day_minutes': avg_drive_time_per_day,
        'avg_locations_per_secondary_day': avg_locations_per_secondary_day,
        'days_with_routes': len(daily_drive_times),
        'secondary_days_count': len(secondary_assignments)
    }


def generate_daily_itineraries(optimization_package: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Generate detailed daily itineraries with location details and route information.
    
    Args:
        optimization_package: Complete optimization results
        
    Returns:
        Dictionary mapping day -> itinerary details
    """
    daily_routes = optimization_package['daily_routes']
    daily_drive_times = optimization_package['daily_drive_times']
    route_geometries = optimization_package.get('route_geometries', {})
    primary_assignments = optimization_package['primary_assignments']
    secondary_assignments = optimization_package['secondary_assignments']
    locations_df = optimization_package['locations_df']
    
    # Create location lookup
    location_lookup = {row['location_id']: row for row in locations_df.to_dicts()}
    
    itineraries = {}
    
    for day in sorted(daily_routes.keys()):
        route = daily_routes[day]
        drive_time = daily_drive_times[day]
        
        itinerary = {
            'day': day,
            'route': route,
            'total_drive_time_minutes': drive_time,
            'total_locations': len(route),
            'locations': []
        }
        
        # Determine day type
        if day in primary_assignments:
            itinerary['day_type'] = 'primary'
            itinerary['primary_location_id'] = primary_assignments[day]
        elif day in secondary_assignments:
            itinerary['day_type'] = 'secondary'
            itinerary['secondary_location_ids'] = secondary_assignments[day]
        
        # Add location details
        for i, location_id in enumerate(route):
            location_info = location_lookup.get(location_id, {})
            location_detail = {
                'order': i + 1,
                'location_id': location_id,
                'name': location_info.get('name', f'Location {location_id}'),
                'address': location_info.get('address', ''),
                'latitude': location_info.get('latitude'),
                'longitude': location_info.get('longitude'),
                'location_class': location_info.get('location_class', 'unknown')
            }
            itinerary['locations'].append(location_detail)
        
        # Add route geometry info if available
        if day in route_geometries and route_geometries[day]:
            route_geom = route_geometries[day]
            itinerary['route_geometry'] = {
                'total_distance_meters': route_geom.total_distance_meters,
                'total_duration_seconds': route_geom.total_duration_seconds,
                'geometry_polyline': route_geom.geometry_polyline,
                'turn_instructions_count': len(route_geom.turn_by_turn_instructions)
            }
        
        itineraries[day] = itinerary
    
    return itineraries


def generate_zone_report_package(optimization_package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zone Report Generator - Compile complete zone optimization report.
    
    Args:
        optimization_package: Complete optimization results
        
    Returns:
        Complete zone report package with routes, metrics, and analytics
    """
    zone_id = optimization_package['zone_id']
    
    logger.info(f"Generating zone report package for {zone_id}")
    
    # Generate core metrics
    metrics_summary = generate_route_metrics_summary(optimization_package)
    
    # Generate detailed itineraries
    daily_itineraries = generate_daily_itineraries(optimization_package)
    
    # Add timestamps and metadata
    report_timestamp = datetime.now()
    
    zone_report = {
        'zone_id': zone_id,
        'report_timestamp': report_timestamp.isoformat(),
        'optimization_metadata': optimization_package.get('metadata', {}),
        'metrics_summary': metrics_summary,
        'daily_itineraries': daily_itineraries,
        'primary_assignments': optimization_package['primary_assignments'],
        'secondary_assignments': optimization_package['secondary_assignments'],
        'daily_routes': optimization_package['daily_routes'],
        'daily_drive_times': optimization_package['daily_drive_times'],
        'config': optimization_package['config']
    }
    
    return zone_report


def print_optimization_solution(optimization_package: Dict[str, Any]) -> None:
    """Print human-readable optimization results to console."""
    zone_report = generate_zone_report_package(optimization_package)
    
    logger.info("Route Optimization Results")
    logger.info("=" * 50)
    
    # Summary statistics
    metrics = zone_report['metrics_summary']
    logger.info(f"Zone ID: {zone_report['zone_id']}")
    logger.info(f"Total locations: {metrics['total_locations_visited']}")
    logger.info(f"Total drive time: {metrics['total_drive_time_minutes']:.1f} minutes")
    logger.info(f"Primary stores: {metrics['primary_store_count']}")
    logger.info(f"Secondary stores: {metrics['non_primary_store_count']}")
    logger.info(f"Unutilized time: {metrics['unutilized_time_hours']:.1f} hours")
    logger.info("")
    
    # Daily schedules
    for day, itinerary in zone_report['daily_itineraries'].items():
        logger.info(f"Day {day} ({itinerary['day_type'].upper()}):")
        
        if itinerary['day_type'] == 'primary':
            primary_location = itinerary['locations'][0]
            logger.info(f"  PRIMARY: {primary_location['name']} (full day)")
            logger.info(f"  Address: {primary_location['address']}")
        else:
            logger.info(f"  SECONDARY ({itinerary['total_locations']} locations):")
            route_names = [loc['name'] for loc in itinerary['locations']]
            logger.info(f"  Route: {' → '.join(route_names)}")
            logger.info(f"  Drive time: {itinerary['total_drive_time_minutes']:.1f} minutes")
            
            # Show route geometry info if available
            if 'route_geometry' in itinerary:
                geom = itinerary['route_geometry']
                logger.info(f"  Route distance: {geom['total_distance_meters']:.0f} meters")
                logger.info(f"  Turn instructions: {geom['turn_instructions_count']}")
        
        logger.info("")


def save_optimization_solution(optimization_package: Dict[str, Any], output_path: str) -> None:
    """Save complete optimization results to JSON file."""
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    zone_report = generate_zone_report_package(optimization_package)
    
    # Convert route geometries to serializable format
    route_geometries_serializable = {}
    for day, route_geom in optimization_package.get('route_geometries', {}).items():
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
    
    # Add serialized route geometries to report
    zone_report['route_geometries'] = route_geometries_serializable
    
    with open(output_path, 'w') as f:
        json.dump(zone_report, f, indent=2, default=str)
    
    logger.info(f"Zone report saved to {output_path}")


def generate_visualization_package(optimization_package: Dict[str, Any], output_dir: str = "output/visualizations") -> Dict[str, str]:
    """Generate interactive maps and visualizations for the optimization results."""
    try:
        zone_id = optimization_package['zone_id']
        locations_df = optimization_package['locations_df']
        daily_routes = optimization_package['daily_routes']
        route_geometries = optimization_package.get('route_geometries', {})
        
        visualization_files = {}
        
        # Generate route map
        route_map = create_route_map(
            locations_df=locations_df,
            daily_routes=daily_routes,
            route_geometries=route_geometries,
            zone_id=zone_id
        )
        
        # Save visualization with itinerary
        viz_path = save_route_visualization(
            route_map=route_map,
            zone_id=zone_id,
            output_dir=output_dir,
            optimization_package=optimization_package
        )
        
        visualization_files['route_map'] = viz_path
        
        logger.info(f"Visualization package generated: {len(visualization_files)} files")
        return visualization_files
        
    except ImportError as e:
        logger.warning(f"Visualization dependencies not available: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    from .stage0_data_ingestion import load_locations_from_jsonl, create_zone_optimization_package
    from .stage4_route_optimization import optimize_all_daily_routes
    import yaml
    
    # Load configuration
    with open("config/model-params.yaml", 'r') as f:
        config = yaml.safe_load(f)['model_params']
    
    zone_id = "test_zone"
    locations_df = load_locations_from_jsonl("data/subway_locations.json", zone_id)
    
    logger.info("Starting complete optimization and reporting pipeline...")
    
    # Create optimization package with data processing
    optimization_package = create_zone_optimization_package(locations_df, config, zone_id)
    
    # Optimize all routes
    optimization_package = optimize_all_daily_routes(optimization_package)
    
    # Generate reports
    zone_report = generate_zone_report_package(optimization_package)
    
    # Print results
    print_optimization_solution(optimization_package)
    
    # Save results
    save_optimization_solution(optimization_package, "output/zone_report.json")
    
    # Generate visualizations
    viz_files = generate_visualization_package(optimization_package)
    
    logger.info(f"Complete reporting pipeline finished")
    logger.info(f"Zone report: {len(zone_report['daily_itineraries'])} daily itineraries")
    logger.info(f"Visualizations: {len(viz_files)} files generated")