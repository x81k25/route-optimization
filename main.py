#!/usr/bin/env python3
"""
Main entry point for route optimization with OSRM integration.
"""

from loguru import logger
from src.core.main_optimizer import RouteOptimizer
from src.utils.visualization import visualize_routes


def main():
    """Run route optimization with OSRM integration."""
    logger.info("Starting Route Optimization Pipeline with OSRM Integration...")
    logger.info("=" * 60)
    
    # Initialize optimizer with zone ID
    optimizer = RouteOptimizer(zone_id="sf_subway_zone")
    
    # Run optimization
    logger.info("Running optimization (this may take a moment for OSRM API calls)...")
    result = optimizer.optimize()
    
    # Display results
    optimizer.print_solution(result)
    
    # Save results
    optimizer.save_solution(result, "output/optimization_result.json")
    
    # Generate visualizations
    logger.info("\n" + "=" * 60)
    logger.info("Generating Route Visualizations...")
    logger.info("=" * 60)
    visualize_routes()


if __name__ == "__main__":
    main()