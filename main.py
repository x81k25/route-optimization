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
from src.core.stage6_orchestrator import (
    load_config,
    optimize_single_zone as orchestrator_optimize_single_zone,
    optimize_multiple_zones
)
from src.core.stage0_data_ingestion import load_locations_from_jsonl
from src.utils.clustering_utils import cluster_locations


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


def run_single_zone_optimization(zone_id: str, locations_path: str = "data/subway_locations.jsonl") -> None:
    """Run optimization for a single zone using new functional architecture."""
    logger.info(f"Starting single-zone optimization for {zone_id}...")
    
    # Load configuration and data
    config = load_config()
    locations_df = load_locations_from_jsonl(locations_path, zone_id)
    
    if len(locations_df) == 0:
        logger.error(f"No locations found for zone {zone_id}")
        return
    
    # Run complete optimization pipeline
    zone_report = orchestrator_optimize_single_zone(
        locations_df=locations_df,
        config=config,
        zone_id=zone_id,
        generate_visualizations=True,
        save_results=True,
        output_dir="output"
    )
    
    # Print summary
    logger.info(f"Completed {zone_id}:")
    logger.info(f"  Locations: {zone_report['metrics_summary']['total_locations_visited']}")
    logger.info(f"  Drive time: {zone_report['metrics_summary']['total_drive_time_minutes']:.1f} minutes")
    logger.info(f"  Primary stores: {zone_report['metrics_summary']['primary_store_count']}")
    logger.info(f"  Secondary stores: {zone_report['metrics_summary']['non_primary_store_count']}")


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
    
    # Filter out locations with null coordinates and set their zone_id to null
    valid_locations = clean_df.filter(
        (pl.col('latitude').is_not_null()) & (pl.col('longitude').is_not_null())
    )
    
    null_locations = clean_df.filter(
        (pl.col('latitude').is_null()) | (pl.col('longitude').is_null())
    ).with_columns(
        pl.lit(None).alias('zone_id'),
        pl.lit('secondary').alias('class')  # Default class for excluded locations
    )
    
    logger.info(f"Loaded {len(clean_df)} locations total:")
    logger.info(f"  - {len(valid_locations)} with valid coordinates (will be clustered)")
    logger.info(f"  - {len(null_locations)} with null coordinates (excluded from clustering, zone_id set to null)")
    
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
    
    logger.info(f"Clustering valid locations with config: {config}")
    if len(valid_locations) > 0:
        clustered_valid_df, quality_metrics = cluster_locations(valid_locations, config)
        
        # Combine clustered valid locations with null locations
        final_df = pl.concat([clustered_valid_df, null_locations], how="vertical")
        
        # Sort by original ID to maintain order
        if 'id' in final_df.columns:
            final_df = final_df.sort('id')
    else:
        logger.warning("No valid locations found for clustering!")
        final_df = null_locations
        quality_metrics = {'n_clusters': 0}
    
    # Write updated locations back to file
    final_df.write_ndjson(locations_path)
    
    logger.info(f"Successfully re-clustered locations:")
    logger.info(f"  Created {quality_metrics['n_clusters']} zones")
    logger.info(f"  Zone sizes: {quality_metrics['min_cluster_size']} - {quality_metrics['max_cluster_size']}")
    logger.info(f"  Average cluster size: {quality_metrics['avg_cluster_size']:.1f}")
    logger.info(f"  Primary stores: {len(final_df.filter(pl.col('class') == 'primary'))}")
    logger.info(f"  Secondary stores: {len(final_df.filter(pl.col('class') == 'secondary'))}")
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
        # Single zone processing using new functional architecture
        run_single_zone_optimization(single_zone, locations_path)
        logger.info("Single zone optimization complete!")
        
    else:
        # Multi-zone concurrent processing using new architecture
        available_zones = get_available_zones(locations_path)
        
        # Limit to first N zones if max_zones specified
        if max_zones is not None:
            available_zones = available_zones[:max_zones]
            logger.info(f"Limited to first {max_zones} zones: {', '.join(available_zones)}")
        else:
            logger.info(f"Found {len(available_zones)} zones to optimize: {', '.join(available_zones)}")
        
        # Load configuration
        config = load_config()
        
        logger.info(f"Running concurrent optimization with up to {max_workers or 'CPU count'} workers...")
        logger.info("This may take several minutes for OSRM API calls...")
        
        # Run multi-zone optimization
        zone_reports = optimize_multiple_zones(
            locations_path=locations_path,
            config=config,
            zone_ids=available_zones,
            max_workers=max_workers,
            generate_visualizations=True,
            save_results=True,
            output_dir="output"
        )
        
        # Print summary
        successful_zones = {k: v for k, v in zone_reports.items() if 'error' not in v}
        failed_zones = {k: v for k, v in zone_reports.items() if 'error' in v}
        
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-ZONE OPTIMIZATION SUMMARY")
        logger.info("=" * 80)
        
        if successful_zones:
            total_locations = sum(r['metrics_summary']['total_locations_visited'] for r in successful_zones.values())
            total_drive_time = sum(r['metrics_summary']['total_drive_time_minutes'] for r in successful_zones.values())
            
            logger.info(f"Successfully optimized: {len(successful_zones)} zones")
            logger.info(f"Total locations: {total_locations}")
            logger.info(f"Total drive time: {total_drive_time:.1f} minutes ({total_drive_time/60:.1f} hours)")
            logger.info(f"Average drive time per zone: {total_drive_time/len(successful_zones):.1f} minutes")
            
            # Show per-zone summary
            logger.info("\nPer-Zone Results:")
            for zone_id in sorted(successful_zones.keys()):
                report = successful_zones[zone_id]
                logger.info(f"  {zone_id}: {report['metrics_summary']['total_locations_visited']} locations, "
                           f"{report['metrics_summary']['total_drive_time_minutes']:.1f} min")
        
        if failed_zones:
            logger.error(f"\nFailed zones ({len(failed_zones)}):")
            for zone_id, report in failed_zones.items():
                logger.error(f"  {zone_id}: {report.get('error', 'Unknown error')}")
        
        logger.info(f"\nResults saved to output/ directory")
        logger.info(f"Multi-zone summary: output/multi_zone_summary.json")
        logger.info(f"Individual zone reports: output/zones/")
        if successful_zones:
            logger.info(f"Visualizations: output/visualizations/")


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
            logger.info("  python main.py --geocode          # Fix coordinates by geocoding all addresses")
            sys.exit(0)
        elif sys.argv[1].startswith("zone_"):
            run_single_zone_optimization(sys.argv[1])
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
        elif sys.argv[1] == "--geocode":
            logger.info("Starting geocoding process to fix coordinates...")
            from src.utils.geocoding_utils import main as geocoding_main
            result = geocoding_main()
            if result == 0:
                logger.info("Geocoding completed successfully!")
            else:
                logger.error("Geocoding failed!")
                sys.exit(result)
        else:
            logger.error(f"Unknown argument: {sys.argv[1]}")
            sys.exit(1)
    else:
        main()