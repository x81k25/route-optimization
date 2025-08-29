#!/usr/bin/env python3
"""
Main entry point for route optimization with OSRM integration.
Supports both single-zone and multi-zone concurrent processing.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

from loguru import logger
from src.core.main_optimizer import RouteOptimizer
from src.utils.visualization import visualize_routes
from src.utils.clustering_utils import add_zone_ids_to_jsonl_dataset, default_cluster_config


def get_available_zones(locations_path: str = "data/subway_locations.jsonl") -> List[str]:
    """Extract all available zone_ids from the dataset."""
    zones = set()
    
    if locations_path.endswith('.jsonl'):
        with open(locations_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    loc_data = json.loads(line)
                    if 'zone_id' in loc_data:
                        zones.add(loc_data['zone_id'])
    else:
        with open(locations_path, 'r') as f:
            data = json.load(f)
        
        locations_key = None
        if 'subway_locations_california' in data:
            locations_key = 'subway_locations_california'
        elif 'subway_locations_san_francisco' in data:
            locations_key = 'subway_locations_san_francisco'
        else:
            raise ValueError("Could not find locations in dataset")
        
        for loc_data in data[locations_key]:
            if 'zone_id' in loc_data:
                zones.add(loc_data['zone_id'])
    
    return sorted(list(zones))


def optimize_single_zone(zone_id: str, locations_path: str = "data/subway_locations.jsonl") -> tuple:
    """Optimize a single zone and return results."""
    try:
        logger.info(f"Starting optimization for {zone_id}")
        
        # Initialize optimizer for this zone
        optimizer = RouteOptimizer(zone_id=zone_id, locations_path=locations_path)
        
        # Run optimization
        result = optimizer.optimize()
        
        # Save individual zone results
        output_dir = Path("output/zones")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        zone_output_path = output_dir / f"{zone_id}_result.json"
        optimizer.save_solution(result, str(zone_output_path))
        
        logger.info(f"Completed {zone_id}: {result.total_locations_visited()} locations, "
                   f"{result.total_drive_time():.1f} min drive time")
        
        return (zone_id, result, None)
        
    except Exception as e:
        logger.error(f"Failed to optimize {zone_id}: {e}")
        return (zone_id, None, str(e))


def run_clustering(
    locations_path: str = "data/subway_locations.jsonl",
    min_locations: int = 3,
    max_locations: int = 15,
    primary_min: int = 0,
    primary_max: int = 3,
    method: str = "kmeans"
):
    """
    Re-cluster locations and assign new zone_ids and primary store status.
    
    Args:
        locations_path: Path to JSONL locations file
        min_locations: Minimum locations per cluster
        max_locations: Maximum locations per cluster
        primary_min: Minimum primary stores per zone
        primary_max: Maximum primary stores per zone
        method: Clustering method ("kmeans" or "geographic")
    """
    import polars as pl
    
    logger.info("Starting location clustering...")
    logger.info(f"Input file: {locations_path}")
    
    # Read existing locations and strip zone information
    logger.info("Reading existing location data...")
    locations_df = pl.read_ndjson(locations_path)
    
    # Remove existing zone_id and class columns if they exist
    columns_to_keep = [col for col in locations_df.columns if col not in ['zone_id', 'class']]
    clean_df = locations_df.select(columns_to_keep)
    
    logger.info(f"Loaded {len(clean_df)} locations, stripped existing zone assignments")
    
    # Create clustering config
    config = {
        'min_locations_per_cluster': min_locations,
        'max_locations_per_cluster': max_locations,
        'method': method,
        'random_seed': 42,
        'primary_store_min': primary_min,
        'primary_store_max': primary_max
    }
    
    # Import and run clustering
    from src.utils.clustering_utils import cluster_locations
    
    logger.info(f"Clustering with config: {config}")
    clustered_df, quality_metrics = cluster_locations(clean_df, config)
    
    # Write updated locations back to file
    clustered_df.write_ndjson(locations_path)
    
    logger.info(f"Successfully re-clustered locations:")
    logger.info(f"  Created {quality_metrics['n_clusters']} zones")
    logger.info(f"  Zone sizes: {quality_metrics['min_cluster_size']} - {quality_metrics['max_cluster_size']}")
    logger.info(f"  Average cluster size: {quality_metrics['avg_cluster_size']:.1f}")
    logger.info(f"  Primary stores: {len(clustered_df.filter(pl.col('class') == 'primary'))}")
    logger.info(f"  Secondary stores: {len(clustered_df.filter(pl.col('class') == 'secondary'))}")
    logger.info(f"Updated {locations_path} with new zone assignments")


def main(single_zone: str = None, max_workers: int = None, max_zones: int = None):
    """
    Run route optimization with OSRM integration.
    
    Args:
        single_zone: If provided, optimize only this zone
        max_workers: Number of parallel processes (default: CPU count)
        max_zones: Limit to first N zones by zone order (default: all zones)
    """
    logger.info("Starting Route Optimization Pipeline with OSRM Integration...")
    logger.info("=" * 80)
    
    locations_path = "data/subway_locations.jsonl"
    
    if single_zone:
        # Single zone processing (original behavior)
        logger.info(f"Running single-zone optimization for {single_zone}...")
        
        optimizer = RouteOptimizer(zone_id=single_zone, locations_path=locations_path)
        result = optimizer.optimize()
        
        # Display results
        optimizer.print_solution(result)
        
        # Save results
        optimizer.save_solution(result, "output/optimization_result.json")
        
        # Generate visualizations
        logger.info("\n" + "=" * 60)
        logger.info("Generating Route Visualizations...")
        logger.info("=" * 60)
        visualize_routes(locations_path=locations_path)
        
    else:
        # Multi-zone concurrent processing
        available_zones = get_available_zones(locations_path)
        
        # Limit to first N zones if max_zones specified
        if max_zones is not None:
            available_zones = available_zones[:max_zones]
            logger.info(f"Limited to first {max_zones} zones: {', '.join(available_zones)}")
        else:
            logger.info(f"Found {len(available_zones)} zones to optimize: {', '.join(available_zones)}")
        
        if max_workers is None:
            max_workers = min(cpu_count(), len(available_zones))
        
        logger.info(f"Running concurrent optimization with {max_workers} workers...")
        logger.info("This may take several minutes for OSRM API calls...")
        
        # Run concurrent optimization
        zone_results = {}
        zone_errors = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all zone optimization tasks
            future_to_zone = {
                executor.submit(optimize_single_zone, zone_id, locations_path): zone_id
                for zone_id in available_zones
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_zone):
                zone_id, result, error = future.result()
                
                if error:
                    zone_errors[zone_id] = error
                else:
                    zone_results[zone_id] = result
        
        # Generate summary report
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-ZONE OPTIMIZATION SUMMARY")
        logger.info("=" * 80)
        
        if zone_results:
            total_locations = sum(r.total_locations_visited() for r in zone_results.values())
            total_drive_time = sum(r.total_drive_time() for r in zone_results.values())
            
            logger.info(f"Successfully optimized: {len(zone_results)} zones")
            logger.info(f"Total locations: {total_locations}")
            logger.info(f"Total drive time: {total_drive_time:.1f} minutes ({total_drive_time/60:.1f} hours)")
            logger.info(f"Average drive time per zone: {total_drive_time/len(zone_results):.1f} minutes")
            
            # Show per-zone summary
            logger.info("\nPer-Zone Results:")
            for zone_id in sorted(zone_results.keys()):
                result = zone_results[zone_id]
                logger.info(f"  {zone_id}: {result.total_locations_visited()} locations, "
                           f"{result.total_drive_time():.1f} min")
        
        if zone_errors:
            logger.error(f"\nFailed zones ({len(zone_errors)}):")
            for zone_id, error in zone_errors.items():
                logger.error(f"  {zone_id}: {error}")
        
        # Combine results for overall visualization (optional)
        if zone_results:
            logger.info("\n" + "=" * 60)
            logger.info("Saving combined results...")
            
            # Save flattened JSONL format for tabular viewing
            with open("output/multi_zone_summary.jsonl", 'w') as f:
                for zone_id in sorted(zone_results.keys()):
                    result = zone_results[zone_id]
                    zone_record = {
                        'zone_id': zone_id,
                        'locations_visited': result.total_locations_visited(),
                        'primary_store_count': result.primary_store_count(),
                        'non_primary_store_count': result.non_primary_store_count(),
                        'total_primary_hours': result.total_primary_hours(),
                        'total_non_primary_hours': result.total_non_primary_hours(),
                        'unutilized_time': result.unutilized_time(),
                        'drive_time_minutes': result.total_drive_time(),
                        'optimization_time_seconds': result.metadata['optimization_duration_seconds']
                    }
                    f.write(json.dumps(zone_record) + '\n')
            
            logger.info("Saved flattened results to output/multi_zone_summary.jsonl")
            logger.info("Individual zone results saved to output/zones/")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            logger.info("Usage:")
            logger.info("  python main.py                    # Run all zones concurrently")
            logger.info("  python main.py zone_001           # Run single zone")
            logger.info("  python main.py --workers 4        # Limit concurrent workers")
            logger.info("  python main.py --zones 5          # Run first 5 zones only")
            logger.info("  python main.py --workers 2 --zones 3  # 2 workers, first 3 zones")
            logger.info("  python main.py --cluster          # Re-cluster locations with default settings")
            logger.info("  python main.py --cluster --min 5 --max 20  # Custom cluster sizes")
            logger.info("  python main.py --cluster --primary-min 1 --primary-max 2  # Custom primary store range")
            logger.info("  python main.py --cluster --method geographic  # Use geographic clustering")
            sys.exit(0)
        elif sys.argv[1].startswith("zone_"):
            main(single_zone=sys.argv[1])
        elif sys.argv[1] == "--workers" and len(sys.argv) > 2:
            if len(sys.argv) > 3 and sys.argv[3] == "--zones":
                main(max_workers=int(sys.argv[2]), max_zones=int(sys.argv[4]))
            else:
                main(max_workers=int(sys.argv[2]))
        elif sys.argv[1] == "--zones" and len(sys.argv) > 2:
            if len(sys.argv) > 3 and sys.argv[3] == "--workers":
                main(max_zones=int(sys.argv[2]), max_workers=int(sys.argv[4]))
            else:
                main(max_zones=int(sys.argv[2]))
        elif sys.argv[1] == "--cluster":
            # Parse clustering arguments
            min_locations = 3
            max_locations = 15
            primary_min = 0
            primary_max = 3
            method = "kmeans"
            
            i = 2
            while i < len(sys.argv):
                if sys.argv[i] == "--min" and i + 1 < len(sys.argv):
                    min_locations = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--max" and i + 1 < len(sys.argv):
                    max_locations = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--primary-min" and i + 1 < len(sys.argv):
                    primary_min = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--primary-max" and i + 1 < len(sys.argv):
                    primary_max = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--method" and i + 1 < len(sys.argv):
                    method = sys.argv[i + 1]
                    i += 2
                else:
                    logger.error(f"Unknown clustering argument: {sys.argv[i]}")
                    sys.exit(1)
            
            run_clustering(
                min_locations=min_locations,
                max_locations=max_locations,
                primary_min=primary_min,
                primary_max=primary_max,
                method=method
            )
        else:
            logger.error(f"Unknown argument: {sys.argv[1]}")
            sys.exit(1)
    else:
        main()