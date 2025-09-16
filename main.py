#!/usr/bin/env python3

# standard library imports
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import List, Optional

# 3rd-party imports
from dotenv import load_dotenv
from loguru import logger
import polars as pl
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
import yaml

# local imports - new pipeline structure
from src.core._1_extraction import extract_locations, validate_locations
from src.core._3_0_optimization import optimize_zone
from src.core._4_reporting import gen_zone_summary, gen_aggregate_summary
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
    logger.info("stage 1: extraction")
    
    # Extract locations
    locations = extract_locations(pos_path, zone_ids)

    if locations.height == 0:
        logger.error("no valid zones given for processing")
        return None

    # Validate data
    validated_locations = validate_locations(locations)

    logger.success(f"stage 1 complete: {len(validated_locations)} locations extracted and validated")
    logger.info(f"sample data:\n{validated_locations.head()}")

    return validated_locations


# ------------------------------------------------------------------------------
# Stage 2: Preprocessing  
# ------------------------------------------------------------------------------

def preprocessing_stage(locations: pl.DataFrame) -> pl.DataFrame:
    """
    Stage 2: Preprocess data for optimization.

    Currently a passthrough placeholder - preprocessing logic moved to optimization stage.

    :param locations: Location DataFrame
    :return: Same DataFrame (passthrough)
    """
    logger.info("stage 2: preprocessing")

    logger.info(f"passing through {len(locations)} locations from {locations['zone_id'].n_unique()} zones")
    logger.success("stage 2 complete: data passed through for optimization")

    return locations


# ------------------------------------------------------------------------------
# Stage 3: Optimization
# ------------------------------------------------------------------------------

def optimization_stage(
    preprocessed_data: pl.DataFrame,
    model_params: dict,
    clusterer: str,
    balancer: str
) -> pl.DataFrame:
    """
    Stage 3: Optimize routes for all zones.

    :param preprocessed_data: DataFrame containing all location data
    :param model_params: Model parameters configuration dictionary
    :param clusterer: Clustering algorithm for secondary locations
    :param balancer: Balancing approach for workload equalization
    :return: Complete itinerary DataFrame
    """
    logger.info("stage 3: optimization")
    
    # Partition data by zone_id
    zone_data = {}
    for zone_id in preprocessed_data["zone_id"].unique().to_list():
        zone_data[zone_id] = preprocessed_data.filter(pl.col("zone_id") == zone_id)

    zone_count = len(zone_data)
    max_workers = max(1, os.cpu_count() // 2)
    logger.info(f"optimizing {zone_count} zone(s) using {max_workers} threads")
    logger.info(f"data partitioned: {[f'{zone_id}({len(df)})' for zone_id, df in zone_data.items()]}")

    # Optimize zones in parallel
    itinerary_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit optimization tasks with zone-specific data
        future_to_zone = {}
        for zone_id, zone_df in zone_data.items():
            future = executor.submit(
                optimize_zone,
                zone_df,  # Pass only zone-specific DataFrame
                zone_id,
                model_params,
                clusterer,
                balancer
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
                    logger.info(f"completed optimization for zone {zone_id}")
                    progress.update(task, advance=1, last_zone=zone_id)
                except Exception as exc:
                    logger.error(f"zone {zone_id} generated an exception: {exc}")
                    raise
    
    # Combine results
    if itinerary_list:
        itinerary = pl.concat(itinerary_list, how='vertical')
        logger.success(f"stage 3 complete: {len(itinerary)} route days optimized")
        logger.info(f"sample results:\n{itinerary.head()}")
        return itinerary
    else:
        logger.warning("no itinerary data generated")
        return pl.DataFrame()


# ------------------------------------------------------------------------------
# Stage 4: Reporting
# ------------------------------------------------------------------------------

def reporting_stage(
    itinerary: pl.DataFrame,
    model_params: dict
) -> tuple:
    """
    Stage 4: Generate analytics and reports from itinerary data.

    :param itinerary: Itinerary DataFrame from optimization stage
    :param model_params: Model configuration parameters
    :return: Tuple of (daily_summary, zone_summary, aggregate_summary)
    """
    logger.info("stage 4: reporting")

    # Generate daily summary from itinerary using new function
    from src.core._4_reporting import gen_daily_summary
    daily_summary = gen_daily_summary(itinerary, model_params)

    if len(daily_summary) == 0:
        logger.warning("no data to report on")
        return None, None, None

    zone_count = daily_summary['zone_id'].n_unique()
    logger.info(f"generating reports for {zone_count} zone(s)")

    # Generate zone summary metrics
    zone_summary = gen_zone_summary(daily_summary, model_params)

    # Generate aggregate summary statistics
    aggregate_summary = gen_aggregate_summary(zone_summary)
    
    logger.success("stage 4 complete: reports generated")
    logger.info(f"aggregate summary:\n{aggregate_summary}")

    return daily_summary, zone_summary, aggregate_summary


# ------------------------------------------------------------------------------
# Stage 5: Loading
# ------------------------------------------------------------------------------

def loading_stage(
    itinerary: pl.DataFrame,
    daily_summary: Optional[pl.DataFrame] = None,
    zone_summary: Optional[pl.DataFrame] = None,
    aggregate_summary: Optional[pl.DataFrame] = None
) -> None:
    """
    Stage 5: Export results to files.

    :param itinerary: Complete itinerary DataFrame (individual position records)
    :param daily_summary: Daily summary DataFrame
    :param zone_summary: Zone-level summary DataFrame
    :param aggregate_summary: Aggregate summary statistics DataFrame
    """
    logger.info("stage 5: loading")

    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)

    # Export all results
    load_results_to_files(
        itinerary=itinerary,
        daily_summary=daily_summary,
        zone_summary=zone_summary,
        aggregate_summary=aggregate_summary,
        output_dir="./output"
    )
    
    logger.success("stage 5 complete: results exported to ./output/")


# ------------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ------------------------------------------------------------------------------

def main(
    zone_ids: Optional[List[str]] = None,
    local: bool = True,
    clusterer: str = "none",
    balancer: str = "none"
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
    logger.info("route optimization pipeline starting")
    logger.info("pipeline structure: extraction → preprocessing → optimization → reporting → loading")

    # Load model parameters configuration
    with open("./config/model-params.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_params = config["model_params"]
    logger.info(f"loaded model parameters: {model_params}")

    try:
        # Stage 1: Extraction
        locations = extraction_stage(zone_ids)
        if locations is None or locations.is_empty():
            logger.error("pipeline terminated: no data extracted")
            return

        # Stage 2: Preprocessing
        preprocessed_data = preprocessing_stage(locations)
        if preprocessed_data.is_empty():
            logger.error("pipeline terminated: no zones preprocessed")
            return

        # Stage 3: Optimization
        itinerary = optimization_stage(preprocessed_data, model_params, clusterer, balancer)
        if len(itinerary) == 0:
            logger.error("pipeline terminated: no routes optimized")
            return

        # Stage 4: Reporting
        daily_summary, zone_summary, aggregate_summary = reporting_stage(itinerary, model_params)

        # Stage 5: Loading
        loading_stage(itinerary, daily_summary, zone_summary, aggregate_summary)
        
        logger.success("✅ ROUTE OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"pipeline failed: {e}")
        raise


def multi_main(
    zone_ids: Optional[List[str]] = None,
    local: bool = True,
    clusterers: List[str] = ["mds_kmeans"],
    balancers: List[str] = ["none"]
) -> None:
    """
    Multi-algorithm orchestrator that runs the pipeline for all combinations
    of clusterer and balancer algorithms.

    :param zone_ids: List of zone_ids to optimize
    :param local: Whether operations use local files
    :param clusterers: List of clustering algorithms to test
    :param balancers: List of balancing approaches to test
    """
    # Create grid of all combinations
    algorithm_grid = list(product(clusterers, balancers))
    total_combinations = len(algorithm_grid)

    logger.info(f"🔀 MULTI-ALGORITHM ORCHESTRATOR STARTING")
    logger.info(f"running {total_combinations} combinations: {clusterers} × {balancers}")

    # Run each combination sequentially (no parallelization as optimization is already parallel)
    for i, (clusterer, balancer) in enumerate(algorithm_grid, 1):
        logger.info(f"🔄 COMBINATION {i}/{total_combinations}: {clusterer} + {balancer}")
        logger.info("-"*60)

        try:
            # Run main pipeline for this combination
            main(
                zone_ids=zone_ids,
                local=local,
                clusterer=clusterer,
                balancer=balancer
            )
            logger.success(f"✅ Combination {i}/{total_combinations} completed: {clusterer} + {balancer}")

        except Exception as e:
            logger.error(f"combination {i}/{total_combinations} failed: {clusterer} + {balancer} - {e}")
            # Continue with next combination rather than stopping entire orchestration
            continue


    logger.success(f"multi-algorithm orchestrator completed")
    logger.info(f"processed {total_combinations} combinations across {len(clusterers)} clusterers and {len(balancers)} balancers")


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
        nargs="*",
        default=["none"],
        choices=["none", "mds_kmeans", "dbscan", "hierarchical", "spectral", "balanced"],
        help="Clustering algorithm(s) for secondary locations (default: none). Multiple values will run all combinations."
    )

    parser.add_argument(
        '--balancer',
        type=str,
        nargs="*",
        default=["none"],
        choices=["none", "greedy", "local_search", "simulated_annealing", "min_max", "network_flow"],
        help="Balancing approach(es) for workload equalization (default: none - no balancing). Multiple values will run all combinations."
    )

    parser.add_argument(
        '--full-grid',
        action='store_true',
        help="Run all possible combinations of all available clusterers and balancers. Overrides --clusterer and --balancer arguments."
    )

    args = parser.parse_args()

    # Configure dual logger setup based on zone argument usage
    main_log_level = os.getenv("MAIN_LOG_LEVEL", "INFO").upper()
    module_log_level = os.getenv("MODULE_LOG_LEVEL", "INFO").upper()

    logger.remove()  # Remove default logger

    # Determine if zones are specified (either specific zones or empty list means "all zones")
    zones_specified = args.zone_ids is not None and len(args.zone_ids) > 0

    if zones_specified:
        # Testing specific zones - use env log levels for both main and core
        logger.add(
            sink=sys.stderr,
            level=main_log_level,
            filter=lambda record: "main.py" in record["file"].name,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | MAIN | <level>{message}</level>"
        )
        logger.add(
            sink=sys.stderr,
            level=module_log_level,
            filter=lambda record: record["name"].startswith("src."),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
        logger.info(f"🔍 Zone-specific run detected: Using {main_log_level} level for main, {module_log_level} level for core loggers")
    else:
        # Full pipeline run - use env levels for main and modules
        logger.add(
            sink=sys.stderr,
            level=main_log_level,
            filter=lambda record: "main.py" in record["file"].name,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | MAIN | <level>{message}</level>"
        )
        logger.add(
            sink=sys.stderr,
            level=module_log_level,
            filter=lambda record: record["name"].startswith("src."),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
        )
        logger.info(f"full pipeline run: using {main_log_level} level for main, {module_log_level} level for core loggers")

    # Handle --full-grid argument
    if args.full_grid:
        # Override arguments with all available options
        all_clusterers = ["none", "mds_kmeans", "dbscan", "hierarchical", "spectral", "balanced"]
        all_balancers = ["none", "greedy", "local_search", "simulated_annealing", "min_max", "network_flow"]

        logger.info(f"full grid mode: running all {len(all_clusterers)} × {len(all_balancers)} = {len(all_clusterers) * len(all_balancers)} combinations")

        # Multi-algorithm orchestration with full grid
        multi_main(
            zone_ids=args.zone_ids,
            local=args.local,
            clusterers=all_clusterers,
            balancers=all_balancers
        )
    else:
        # Determine if we should run single or multi execution
        is_multi_execution = len(args.clusterer) > 1 or len(args.balancer) > 1

        if is_multi_execution:
            # Multi-algorithm orchestration
            multi_main(
                zone_ids=args.zone_ids,
                local=args.local,
                clusterers=args.clusterer,
                balancers=args.balancer
            )
        else:
            # Single algorithm execution
            main(
                zone_ids=args.zone_ids,
                local=args.local,
                clusterer=args.clusterer[0],  # Extract single value from list
                balancer=args.balancer[0]     # Extract single value from list
            )


# ------------------------------------------------------------------------------
# End of main.py - 5 Stage Pipeline Architecture
# ------------------------------------------------------------------------------