#!/usr/bin/env python3
"""
Convert HTML visualizations to PNG format.
Run this after installing Chrome: plotly_get_chrome
"""

import os
import glob
from pathlib import Path
from src.utils.visualization import RouteVisualizer


def convert_html_to_png(output_dir: str = "output"):
    """Convert all HTML files in output directory to PNG."""
    
    try:
        # Test if Chrome is available
        import plotly.graph_objects as go
        test_fig = go.Figure()
        test_fig.write_image("test.png", width=100, height=100)
        os.remove("test.png")
        print("Chrome is available, proceeding with PNG conversion...")
        
    except Exception as e:
        print(f"Chrome not available for PNG export: {e}")
        print("Run 'plotly_get_chrome' to install Chrome, then try again.")
        return
    
    # Find all HTML files
    html_files = glob.glob(f"{output_dir}/*.html")
    
    if not html_files:
        print(f"No HTML files found in {output_dir}/")
        return
    
    print(f"Converting {len(html_files)} HTML files to PNG...")
    
    # Re-generate with PNG export
    visualizer = RouteVisualizer()
    
    # Load results and regenerate
    results_path = f"{output_dir}/optimization_result.json"
    if os.path.exists(results_path):
        visualizer.generate_all_visualizations(results_path, output_dir)
        print("PNG conversion completed!")
    else:
        print(f"Optimization results not found: {results_path}")


if __name__ == "__main__":
    convert_html_to_png()