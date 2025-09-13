#!/usr/bin/env python3

# standard library imports
import argparse
import os
import sys
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 3rd-party imports
from loguru import logger
import polars as pl
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

# local imports - new pipeline structure
from src.core._1_extraction import extract_locations, validate_locations
from src.core._2_preprocessing import preprocess_zone_data, group_by_zones
from src.core._3_0_optimization import optimize_zone
from src.core._4_reporting import aggregate as generate_reports
from src.core._5_loading import load_results_to_files


# ------------------------------------------------------------------------------
# Stage 1: Extraction
# ------------------------------------------------------------------------------

def extraction_stage(
    zone_ids: Optional[List[str]] = None,
    pos_path: str = "./data/locations.jsonl"
) -> pl.DataFrame:
    """
    Stage 1: Extract and validate location data.
    
    :param zone_ids: List of zone_ids to extract
    :param pos_path: Path to locations file
    :return: Validated location DataFrame
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: EXTRACTION")
    logger.info("=" * 60)
    
    # Extract locations
    pos_df = extract_locations(pos_path, zone_ids)
    
    if pos_df.height == 0:
        logger.error("No valid zones given for processing")
        return None
    
    # Validate data
    validated_df = validate_locations(pos_df)
    
    logger.success(f"Stage 1 complete: {len(validated_df)} locations extracted and validated")
    logger.info(f"Sample data:\n{validated_df.head()}")
    
    return validated_df


# ------------------------------------------------------------------------------
# Stage 2: Preprocessing  
# ------------------------------------------------------------------------------

def preprocessing_stage(pos_df: pl.DataFrame) -> dict:
    """
    Stage 2: Preprocess data for optimization.
    
    :param pos_df: Location DataFrame
    :return: Dictionary with preprocessed zone data
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: PREPROCESSING")
    logger.info("=" * 60)
    
    # Group by zones
    zone_groups = group_by_zones(pos_df)
    
    # Preprocess each zone with Rich progress bar
    zone_data = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Preprocessing zones"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("[dim]Current: {task.fields[current_zone]}"),
        expand=True
    ) as progress:
        task = progress.add_task("preprocessing", total=len(zone_groups), current_zone="Starting...")
        
        for zone_id, zone_df in zone_groups.items():
            progress.update(task, current_zone=zone_id)
            zone_df, centroid, od_matrix = preprocess_zone_data(zone_df, zone_id)
            zone_data[zone_id] = {
                'df': zone_df,
                'centroid': centroid, 
                'od_matrix': od_matrix
            }
            progress.advance(task)
    
    logger.success(f"Stage 2 complete: {len(zone_data)} zones preprocessed")
    
    return zone_data


# ------------------------------------------------------------------------------
# Stage 3: Optimization
# ------------------------------------------------------------------------------

def optimization_stage(zone_data: dict, clusterer: str = "mds_kmeans", balancer: str = "greedy") -> pl.DataFrame:
    """
    Stage 3: Optimize routes for all zones.
    
    :param zone_data: Preprocessed zone data
    :param clusterer: Clustering algorithm for secondary locations
    :param balancer: Balancing approach for workload equalization
    :return: Complete itinerary DataFrame
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: OPTIMIZATION")
    logger.info("=" * 60)
    
    zone_count = len(zone_data)
    max_workers = max(1, os.cpu_count() // 2)
    logger.info(f"Optimizing {zone_count} zone(s) using {max_workers} threads")
    
    # Optimize zones in parallel
    itinerary_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit optimization tasks
        future_to_zone = {}
        for zone_id, data in zone_data.items():
            future = executor.submit(
                optimize_zone, 
                data['df'], 
                zone_id, 
                data['od_matrix'],
                clusterer,
                balancer,
                data['centroid']
            )
            future_to_zone[future] = zone_id
        
        # Collect results with Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Optimizing zones"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]Last: {task.fields[last_zone]}"),
            expand=True
        ) as progress:
            task = progress.add_task("optimization", total=len(future_to_zone), last_zone="None")
            
            for future in as_completed(future_to_zone):
                zone_id = future_to_zone[future]
                try:
                    itinerary_zone = future.result()
                    if len(itinerary_zone) > 0:
                        itinerary_list.append(itinerary_zone)
                    logger.info(f"Completed optimization for zone {zone_id}")
                    progress.update(task, advance=1, last_zone=zone_id)
                except Exception as exc:
                    logger.error(f"Zone {zone_id} generated an exception: {exc}")
                    raise
    
    # Combine results
    if itinerary_list:
        itinerary = pl.concat(itinerary_list, how='vertical')
        logger.success(f"Stage 3 complete: {len(itinerary)} route days optimized")
        logger.info(f"Sample results:\n{itinerary.head()}")
        return itinerary
    else:
        logger.warning("No itinerary data generated")
        return pl.DataFrame()


# ------------------------------------------------------------------------------
# Stage 4: Reporting
# ------------------------------------------------------------------------------

def reporting_stage(itinerary_df: pl.DataFrame) -> tuple:
    """
    Stage 4: Generate analytics and reports.
    
    :param itinerary_df: Complete itinerary DataFrame
    :return: Tuple of (aggregate_df, summary_df)
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: REPORTING")
    logger.info("=" * 60)
    
    if len(itinerary_df) == 0:
        logger.warning("No data to report on")
        return None, None
    
    zone_count = itinerary_df['zone_id'].n_unique()
    logger.info(f"Generating reports for {zone_count} zone(s)")
    
    # Generate aggregate metrics
    aggregate_df = generate_reports(itinerary_df, local=True)
    
    # Generate summary statistics
    summary_df = aggregate_df.select([
        pl.col('weekly_duration').mean().alias("average_weekly_duration"),
        pl.col('utilization').mean().alias("average_utilization"),
        pl.col('overutilized_days').mean().alias("average_overutilized_days"),
        pl.col('underutilized_days').mean().alias("average_underutilized_days"),
        pl.col('total_pos_time').mean().alias("average_daily_pos_time"),
        pl.col('total_drive_time').mean().alias("average_daily_drive_time"),  
        pl.col('sec_std').mean().alias("average_secondary_duration_standard_deviation"),  
    ])
    
    logger.success("Stage 4 complete: Reports generated")
    logger.info(f"Aggregate summary:\n{summary_df}")
    
    return aggregate_df, summary_df


# ------------------------------------------------------------------------------
# Stage 5: Loading
# ------------------------------------------------------------------------------

def loading_stage(
    itinerary_df: pl.DataFrame,
    aggregate_df: Optional[pl.DataFrame] = None,
    summary_df: Optional[pl.DataFrame] = None
) -> None:
    """
    Stage 5: Export results to files.
    
    :param itinerary_df: Complete itinerary DataFrame
    :param aggregate_df: Aggregate analytics DataFrame
    :param summary_df: Summary statistics DataFrame
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: LOADING")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)
    
    # Export all results
    load_results_to_files(
        itinerary_df=itinerary_df,
        aggregate_df=aggregate_df,
        summary_df=summary_df,
        output_dir="./output"
    )
    
    logger.success("Stage 5 complete: Results exported to ./output/")


# ------------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ------------------------------------------------------------------------------

def main(
    zone_ids: Optional[List[str]] = None,
    local: bool = True,
    clusterer: str = "mds_kmeans",
    balancer: str = "greedy"
) -> None:
    """
    Main pipeline orchestrator following the 5-stage structure:
    
    1. Extraction - Load and validate location data
    2. Preprocessing - Clean and prepare data
    3. Optimization - Execute route optimization
        3.1. Primary day assignment
        3.2. Secondary day clustering  
        3.3. Route optimization
        3.4. Cluster balancing
        3.5. Detailed routing
    4. Reporting - Generate analytics
    5. Loading - Export results
    
    :param zone_ids: List of zone_ids to optimize
    :param local: Whether operations use local files
    :param clusterer: Clustering algorithm for secondary locations
    :param balancer: Balancing approach for workload equalization
    """
    logger.info("🚀 ROUTE OPTIMIZATION PIPELINE STARTING")
    logger.info("Pipeline Structure: Extraction → Preprocessing → Optimization → Reporting → Loading")
    
    try:
        # Stage 1: Extraction
        pos_df = extraction_stage(zone_ids)
        if pos_df is None or pos_df.is_empty():
            logger.error("Pipeline terminated: No data extracted")
            return
        
        # Stage 2: Preprocessing  
        zone_data = preprocessing_stage(pos_df)
        if not zone_data:
            logger.error("Pipeline terminated: No zones preprocessed")
            return
        
        # Stage 3: Optimization
        itinerary_df = optimization_stage(zone_data, clusterer, balancer)
        if len(itinerary_df) == 0:
            logger.error("Pipeline terminated: No routes optimized")
            return
        
        # Stage 4: Reporting
        aggregate_df, summary_df = reporting_stage(itinerary_df)
        
        # Stage 5: Loading
        loading_stage(itinerary_df, aggregate_df, summary_df)
        
        logger.success("✅ ROUTE OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description='Route Optimization Pipeline - 5 Stage Architecture'
    )
    
    parser.add_argument(
        '-z', 
        '--zone_ids', 
        nargs="*",
        default=None,
        type=str, 
        help='One or more zone_ids to process (default: None - processes all zones)'
    )
    
    parser.add_argument(
        '-l',
        '--local',
        action='store_true',
        default=True,
        help="Enable local file processing (default: True)"
    )
    
    parser.add_argument(
        '--clusterer',
        type=str,
        default="mds_kmeans",
        choices=["mds_kmeans", "dbscan", "hierarchical", "spectral", "balanced"],
        help="Clustering algorithm for secondary locations (default: mds_kmeans)"
    )
    
    parser.add_argument(
        '--balancer',
        type=str,
        default="greedy",
        choices=["greedy", "local_search", "simulated_annealing", "min_max", "network_flow"],
        help="Balancing approach for workload equalization (default: greedy)"
    )

    args = parser.parse_args()

    # Configure dual logger setup based on zone argument usage
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logger.remove()  # Remove default logger
    
    # Determine if zones are specified (either specific zones or empty list means "all zones")
    zones_specified = args.zone_ids is not None and len(args.zone_ids) > 0
    
    if zones_specified:
        # Testing specific zones - use env log level for both main and core
        logger.add(
            sink=sys.stderr,
            level=log_level,
            filter=lambda record: "main.py" in record["file"].name,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | MAIN | <level>{message}</level>"
        )
        logger.add(
            sink=sys.stderr,
            level=log_level,
            filter=lambda record: record["name"].startswith("src."),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
        logger.info(f"🔍 Zone-specific run detected: Using {log_level} level for both main and core loggers")
    else:
        # Full pipeline run - use env level for main, ERROR only for core
        logger.add(
            sink=sys.stderr,
            level=log_level,
            filter=lambda record: "main.py" in record["file"].name,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | MAIN | <level>{message}</level>"
        )
        logger.add(
            sink=sys.stderr,
            level="ERROR",
            filter=lambda record: record["name"].startswith("src."),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
        )
        logger.info(f"🚀 Full pipeline run: Using {log_level} level for main, ERROR level for core loggers")

    main(
        zone_ids=args.zone_ids,
        local=args.local,
        clusterer=args.clusterer,
        balancer=args.balancer
    )


# ------------------------------------------------------------------------------
# End of main.py - 5 Stage Pipeline Architecture
# ------------------------------------------------------------------------------