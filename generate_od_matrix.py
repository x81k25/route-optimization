#!/usr/bin/env python3
"""
Generate simulated OD (Origin-Destination) matrix for route optimization.
Creates drive times between all secondary Subway locations in San Francisco.
"""

import json
import pickle
import numpy as np
import polars as pl
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    return 6371 * c


def simulate_drive_time(distance_km, base_speed_kmh=25):
    """
    Simulate drive time based on distance with SF-specific factors.
    
    Args:
        distance_km: Haversine distance in kilometers
        base_speed_kmh: Base driving speed (SF city driving ~25 km/h)
    
    Returns:
        Drive time in minutes
    """
    # Base time from distance
    base_time_minutes = (distance_km / base_speed_kmh) * 60
    
    # SF-specific factors
    # Hills and traffic add 20-40% overhead
    hill_factor = np.random.uniform(1.2, 1.4)
    
    # One-way streets and turns add 10-20% overhead  
    street_factor = np.random.uniform(1.1, 1.2)
    
    # Some routes have highway access (faster)
    if distance_km > 5:
        highway_chance = 0.3  # 30% chance of highway route
        if np.random.random() < highway_chance:
            highway_factor = 0.8  # 20% faster
        else:
            highway_factor = 1.0
    else:
        highway_factor = 1.0
    
    # Calculate final drive time
    drive_time = base_time_minutes * hill_factor * street_factor * highway_factor
    
    # Add some random noise (±2 minutes)
    noise = np.random.uniform(-2, 2)
    drive_time += noise
    
    # Minimum drive time of 3 minutes
    return max(3.0, round(drive_time, 1))


def load_secondary_locations():
    """Load secondary locations from subway_locations.json."""
    with open('data/subway_locations.json', 'r') as f:
        data = json.load(f)
    
    # Filter to secondary locations only
    secondary_locations = [
        loc for loc in data['subway_locations_san_francisco'] 
        if loc['class'] == 'secondary'
    ]
    
    return secondary_locations


def generate_od_matrix():
    """Generate OD matrix with simulated drive times."""
    # Load locations
    locations = load_secondary_locations()
    n_locations = len(locations)
    
    print(f"Generating OD matrix for {n_locations} secondary locations...")
    
    # Initialize data for polars DataFrame
    origin_ids = []
    destination_ids = []
    origin_names = []
    destination_names = []
    distances_km = []
    drive_times_min = []
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Calculate pairwise distances and drive times
    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            origin_ids.append(origin['id'])
            destination_ids.append(destination['id'])
            origin_names.append(origin['name'])
            destination_names.append(destination['name'])
            
            if i == j:
                # Same location
                distances_km.append(0.0)
                drive_times_min.append(0.0)
            else:
                # Calculate haversine distance
                distance = haversine_distance(
                    origin['latitude'], origin['longitude'],
                    destination['latitude'], destination['longitude']
                )
                distances_km.append(round(distance, 2))
                
                # Simulate drive time
                drive_time = simulate_drive_time(distance)
                drive_times_min.append(drive_time)
    
    # Create polars DataFrame
    od_matrix = pl.DataFrame({
        'origin_id': origin_ids,
        'destination_id': destination_ids,
        'origin_name': origin_names,
        'destination_name': destination_names,
        'distance_km': distances_km,
        'drive_time_minutes': drive_times_min
    })
    
    return od_matrix


def main():
    """Generate and save OD matrix."""
    # Generate OD matrix
    od_matrix = generate_od_matrix()
    
    # Display summary statistics
    print(f"\nOD Matrix Summary:")
    print(f"Total pairs: {len(od_matrix)}")
    print(f"Unique locations: {od_matrix['origin_id'].n_unique()}")
    print(f"Average drive time: {od_matrix['drive_time_minutes'].mean():.1f} minutes")
    print(f"Max drive time: {od_matrix['drive_time_minutes'].max():.1f} minutes")
    print(f"Min drive time (excluding 0): {od_matrix.filter(pl.col('drive_time_minutes') > 0)['drive_time_minutes'].min():.1f} minutes")
    
    # Save as pickle
    with open('data/od_matrix.pkl', 'wb') as f:
        pickle.dump(od_matrix, f)
    
    print(f"\nOD matrix saved to data/od_matrix.pkl")
    
    # Display sample of the data
    print(f"\nSample data:")
    print(od_matrix.head(10))


if __name__ == "__main__":
    main()