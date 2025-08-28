#!/usr/bin/env python3
"""
Main entry point for route optimization.
"""

from src.core.main_optimizer import RouteOptimizer
from src.utils.visualization import visualize_routes


def main():
    """Run route optimization."""
    print("Starting Route Optimization Pipeline...")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = RouteOptimizer()
    
    # Run optimization
    result = optimizer.optimize()
    
    # Display results
    optimizer.print_solution(result)
    
    # Save results
    optimizer.save_solution(result, "output/optimization_result.json")
    
    # Generate visualizations
    print("\n" + "=" * 50)
    print("Generating Route Visualizations...")
    print("=" * 50)
    visualize_routes()


if __name__ == "__main__":
    main()