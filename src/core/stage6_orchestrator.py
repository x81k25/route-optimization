"""
Main Route Optimization Orchestrator

Functional programming implementation that coordinates all processing stages
from the architecture:
- Stage 0: Data Ingestion 
- Stage 1: OSRM OD Matrix Generation
- Stage 2: Primary Location Assignment  
- Stage 3: Secondary Location Clustering
- Stage 4: Individual Day Route Optimization
- Stage 5: Detailed Route Geometry Fetch
- Stage 6: Route Metrics Summary Generation
- Zone Report Generation
- Visualization Generation
"""

import yaml
import polars as pl
from typing import Dict, List, Tuple, Any
from datetime import datetime
from loguru import logger

from .stage0_data_ingestion import (
    load_locations_from_jsonl,
    create_zone_optimization_package
)
from .stage4_route_optimization import optimize_all_daily_routes
from .stage5_reporting import (
    generate_zone_report_package,
    print_optimization_solution,
    save_optimization_solution,
    generate_visualization_package
)


def load_config(config_path: str = "config/model-params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['model_params']


def optimize_single_zone(
    locations_df: pl.DataFrame, 
    config: Dict[str, Any], 
    zone_id: str,
    generate_visualizations: bool = True,
    save_results: bool = True,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Complete end-to-end optimization for a single zone.
    
    Orchestrates all stages from data processing through reporting.
    
    Args:
        locations_df: DataFrame containing location data
        config: Configuration dictionary
        zone_id: Zone identifier
        generate_visualizations: Whether to create maps and visualizations
        save_results: Whether to save results to files
        output_dir: Directory for output files
        
    Returns:
        Complete zone report package
    """
    start_time = datetime.now()
    logger.info(f"Starting end-to-end optimization for zone {zone_id}")
    
    # Stages 0-3: Data Processing (ingestion, OD matrix, assignments)
    optimization_package = create_zone_optimization_package(locations_df, config, zone_id)
    
    # Stage 4: Route Optimization
    optimization_package = optimize_all_daily_routes(optimization_package)
    
    # Stages 5-6: Reporting and Analytics
    zone_report = generate_zone_report_package(optimization_package)
    
    # Add orchestration metadata
    end_time = datetime.now()
    orchestration_metadata = {
        'orchestration_start_time': start_time.isoformat(),
        'orchestration_end_time': end_time.isoformat(),
        'orchestration_duration_seconds': (end_time - start_time).total_seconds(),
        'stages_completed': ['data_processing', 'route_optimization', 'reporting'],
        'zone_id': zone_id
    }
    zone_report['orchestration_metadata'] = orchestration_metadata
    
    # Optional: Generate visualizations
    visualization_files = {}
    if generate_visualizations:
        try:
            visualization_files = generate_visualization_package(optimization_package, f"{output_dir}/visualizations")
            zone_report['visualization_files'] = visualization_files
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
    
    # Optional: Save results
    if save_results:
        output_path = f"{output_dir}/zones/{zone_id}_complete_report.json"
        save_optimization_solution(optimization_package, output_path)
        zone_report['saved_to'] = output_path
    
    logger.info(f"Zone {zone_id} optimization completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    return zone_report


def optimize_multiple_zones(
    locations_path: str,
    config: Dict[str, Any],
    zone_ids: List[str] = None,
    max_workers: int = None,
    generate_visualizations: bool = True,
    save_results: bool = True,
    output_dir: str = "output"
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize multiple zones concurrently.
    
    Args:
        locations_path: Path to locations file (JSON/JSONL)
        config: Configuration dictionary
        zone_ids: List of zone IDs to process (if None, processes all zones)
        max_workers: Maximum number of concurrent workers
        generate_visualizations: Whether to create maps and visualizations
        save_results: Whether to save results to files
        output_dir: Directory for output files
        
    Returns:
        Dictionary mapping zone_id -> zone_report
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count
    import os
    
    # Create output directory
    os.makedirs(f"{output_dir}/zones", exist_ok=True)
    if generate_visualizations:
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # Load all locations data
    all_locations_df = load_locations_from_jsonl(locations_path)
    
    # Get unique zone IDs
    available_zones = all_locations_df['zone_id'].unique().to_list()
    if zone_ids is None:
        zone_ids = available_zones
    else:
        # Filter to only available zones
        zone_ids = [z for z in zone_ids if z in available_zones]
    
    logger.info(f"Processing {len(zone_ids)} zones: {zone_ids}")
    
    # Set up concurrent processing
    if max_workers is None:
        max_workers = min(cpu_count(), len(zone_ids))
    
    zone_reports = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all zone optimization tasks
        future_to_zone = {}
        for zone_id in zone_ids:
            zone_df = all_locations_df.filter(pl.col('zone_id') == zone_id)
            if len(zone_df) > 0:
                future = executor.submit(
                    optimize_single_zone,
                    zone_df, 
                    config, 
                    zone_id,
                    generate_visualizations,
                    save_results,
                    output_dir
                )
                future_to_zone[future] = zone_id
            else:
                logger.warning(f"No locations found for zone {zone_id}")
        
        # Collect results
        for future in as_completed(future_to_zone):
            zone_id = future_to_zone[future]
            try:
                zone_report = future.result()
                zone_reports[zone_id] = zone_report
                logger.info(f"Zone {zone_id} completed successfully")
            except Exception as e:
                logger.error(f"Zone {zone_id} failed: {e}")
                zone_reports[zone_id] = {'error': str(e), 'zone_id': zone_id}
    
    # Generate multi-zone summary
    summary_report = generate_multi_zone_summary(zone_reports)
    if save_results:
        summary_path = f"{output_dir}/multi_zone_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        logger.info(f"Multi-zone summary saved to {summary_path}")
    
    return zone_reports


def generate_multi_zone_summary(zone_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics across multiple zones."""
    successful_zones = {k: v for k, v in zone_reports.items() if 'error' not in v}
    failed_zones = {k: v for k, v in zone_reports.items() if 'error' in v}
    
    if not successful_zones:
        return {
            'total_zones': len(zone_reports),
            'successful_zones': 0,
            'failed_zones': len(failed_zones),
            'error': 'No zones completed successfully'
        }
    
    # Aggregate metrics across successful zones
    total_locations = sum(r['metrics_summary']['total_locations_visited'] for r in successful_zones.values())
    total_drive_time = sum(r['metrics_summary']['total_drive_time_minutes'] for r in successful_zones.values())
    total_primary_stores = sum(r['metrics_summary']['primary_store_count'] for r in successful_zones.values())
    total_secondary_stores = sum(r['metrics_summary']['non_primary_store_count'] for r in successful_zones.values())
    
    avg_drive_time_per_zone = total_drive_time / len(successful_zones)
    avg_locations_per_zone = total_locations / len(successful_zones)
    
    return {
        'summary_timestamp': datetime.now().isoformat(),
        'total_zones': len(zone_reports),
        'successful_zones': len(successful_zones),
        'failed_zones': len(failed_zones),
        'aggregate_metrics': {
            'total_locations_all_zones': total_locations,
            'total_primary_stores': total_primary_stores,
            'total_secondary_stores': total_secondary_stores,
            'total_drive_time_minutes': total_drive_time,
            'total_drive_time_hours': total_drive_time / 60.0,
            'avg_drive_time_per_zone_minutes': avg_drive_time_per_zone,
            'avg_locations_per_zone': avg_locations_per_zone
        },
        'successful_zone_ids': list(successful_zones.keys()),
        'failed_zone_ids': list(failed_zones.keys())
    }


if __name__ == "__main__":
    # Example usage for single zone
    config = load_config()
    
    # Single zone optimization
    zone_id = "test_zone"
    locations_df = load_locations_from_jsonl("data/subway_locations.json", zone_id)
    
    logger.info("Starting single zone optimization...")
    zone_report = optimize_single_zone(locations_df, config, zone_id)
    
    # Print results
    print_optimization_solution({
        'zone_id': zone_id,
        'config': config,
        'locations_df': locations_df,
        **zone_report
    })
    
    logger.info(f"Single zone optimization completed")
    logger.info(f"Total locations: {zone_report['metrics_summary']['total_locations_visited']}")
    logger.info(f"Total drive time: {zone_report['metrics_summary']['total_drive_time_minutes']:.1f} minutes")