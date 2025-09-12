"""
Stage 5: Loading
Export optimized routes and results to output files
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import json


def load_results_to_files(
    itinerary_df: pl.DataFrame,
    aggregate_df: Optional[pl.DataFrame] = None,
    summary_df: Optional[pl.DataFrame] = None,
    output_dir: str = "./output"
) -> None:
    """
    Export all results to output files.
    
    This is stage 5 where we:
    1. Write optimized routes to JSON files
    2. Generate HTML visualizations
    3. Create CSV exports for external systems
    4. Organize output directory structure
    
    Args:
        itinerary_df: Detailed route itineraries
        aggregate_df: Aggregate analytics data
        summary_df: Summary statistics
        output_dir: Output directory path
    """
    logger.info("Stage 5: LOADING - Exporting results to files")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export itinerary data
    export_itinerary_data(itinerary_df, output_path)
    
    # Export aggregate reports if provided
    if aggregate_df is not None:
        export_aggregate_data(aggregate_df, output_path)
    
    # Export summary data if provided  
    if summary_df is not None:
        export_summary_data(summary_df, output_path)
    
    logger.success(f"Results exported to {output_dir}")


def export_itinerary_data(df: pl.DataFrame, output_path: Path) -> None:
    """
    Export detailed itinerary data to JSONL file.
    
    Args:
        df: Itinerary DataFrame
        output_path: Output directory path
    """
    if len(df) == 0:
        logger.warning("No itinerary data to export")
        return
    
    output_file = output_path / "itinerary.jsonl"
    
    # Convert DataFrame to records and write as JSONL
    records = df.to_dicts()
    
    with open(output_file, 'w') as f:
        for record in records:
            # Convert any numpy types to Python types for JSON serialization
            cleaned_record = clean_record_for_json(record)
            f.write(json.dumps(cleaned_record) + '\n')
    
    logger.info(f"Exported {len(df)} itinerary records to {output_file}")


def export_aggregate_data(df: pl.DataFrame, output_path: Path) -> None:
    """
    Export aggregate analytics data to JSONL file.
    
    Args:
        df: Aggregate DataFrame
        output_path: Output directory path
    """
    if len(df) == 0:
        logger.warning("No aggregate data to export")
        return
    
    output_file = output_path / "aggregate-report.jsonl"
    
    records = df.to_dicts()
    
    with open(output_file, 'w') as f:
        for record in records:
            cleaned_record = clean_record_for_json(record)
            f.write(json.dumps(cleaned_record) + '\n')
    
    logger.info(f"Exported {len(df)} aggregate records to {output_file}")


def export_summary_data(df: pl.DataFrame, output_path: Path) -> None:
    """
    Export summary statistics to JSONL file.
    
    Args:
        df: Summary DataFrame
        output_path: Output directory path
    """
    if len(df) == 0:
        logger.warning("No summary data to export")
        return
    
    output_file = output_path / "aggregate-summary.jsonl"
    
    records = df.to_dicts()
    
    with open(output_file, 'w') as f:
        for record in records:
            cleaned_record = clean_record_for_json(record)
            f.write(json.dumps(cleaned_record) + '\n')
    
    logger.info(f"Exported {len(df)} summary records to {output_file}")


def export_csv_files(
    itinerary_df: pl.DataFrame,
    output_path: Path,
    flatten_routes: bool = True
) -> None:
    """
    Export data in CSV format for external systems.
    
    Args:
        itinerary_df: Itinerary DataFrame
        output_path: Output directory path
        flatten_routes: Whether to flatten route geometry
    """
    if len(itinerary_df) == 0:
        logger.warning("No data to export as CSV")
        return
    
    # Create simplified version for CSV export
    csv_df = itinerary_df.select([
        "zone_id",
        "day", 
        "duration"
    ])
    
    # Add location count
    csv_df = csv_df.with_columns(
        pl.col("pos_id").list.len().alias("location_count")
    )
    
    output_file = output_path / "routes_summary.csv"
    csv_df.write_csv(output_file)
    
    logger.info(f"Exported CSV summary to {output_file}")


def clean_record_for_json(record: Dict) -> Dict:
    """
    Clean a record for JSON serialization by converting numpy types.
    
    Args:
        record: Dictionary record
        
    Returns:
        Cleaned record with JSON-serializable types
    """
    cleaned = {}
    
    for key, value in record.items():
        # Handle numpy types and convert to Python types
        if hasattr(value, 'item'):  # numpy scalars
            cleaned[key] = value.item()
        elif isinstance(value, list):
            cleaned[key] = [clean_list_item(item) for item in value]
        else:
            cleaned[key] = value
    
    return cleaned


def clean_list_item(item):
    """
    Clean individual list items for JSON serialization.
    
    Args:
        item: List item to clean
        
    Returns:
        JSON-serializable item
    """
    if hasattr(item, 'item'):  # numpy scalar
        return item.item()
    elif isinstance(item, list):
        return [clean_list_item(subitem) for subitem in item]
    else:
        return item


def create_visualization_files(
    itinerary_df: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Create HTML visualization files for routes.
    
    Args:
        itinerary_df: Itinerary DataFrame
        output_path: Output directory path
    """
    vis_path = output_path / "visualizations"
    vis_path.mkdir(exist_ok=True)
    
    # Create visualizations for each zone
    zones = itinerary_df["zone_id"].unique().to_list()
    
    for zone_id in zones:
        zone_df = itinerary_df.filter(pl.col("zone_id") == zone_id)
        
        if len(zone_df) > 0:
            vis_file = vis_path / f"route_map_{zone_id}.html"
            create_zone_visualization(zone_df, vis_file, zone_id)


def create_zone_visualization(
    zone_df: pl.DataFrame,
    output_file: Path,
    zone_id: str
) -> None:
    """
    Create HTML visualization for a single zone.
    
    Args:
        zone_df: DataFrame for the zone
        output_file: Output HTML file path
        zone_id: Zone identifier
    """
    # This would typically use a mapping library like Folium
    # For now, create a simple HTML placeholder
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Map - {zone_id}</title>
    </head>
    <body>
        <h1>Route Optimization - {zone_id}</h1>
        <p>Total days: {len(zone_df)}</p>
        <p>Interactive map visualization would be generated here using the route geometry data.</p>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created visualization placeholder for {zone_id} at {output_file}")