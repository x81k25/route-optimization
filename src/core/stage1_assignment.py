"""
Stage 1: Day Assignment Algorithm

Assigns secondary locations to available days using drive time-based clustering.
"""

import pickle
import numpy as np
import polars as pl
from typing import List, Dict, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class DayAssignmentOptimizer:
    """Assigns secondary locations to days using hierarchical clustering."""
    
    def __init__(self, od_matrix_path: str = "data/od_matrix.pkl"):
        """
        Initialize with OD matrix.
        
        Args:
            od_matrix_path: Path to pickled polars DataFrame with drive times
        """
        with open(od_matrix_path, 'rb') as f:
            self.od_matrix = pickle.load(f)
        
        # Get unique secondary location IDs
        self.location_ids = sorted(self.od_matrix['origin_id'].unique().to_list())
        self.n_locations = len(self.location_ids)
        
        # Create distance matrix for clustering
        self.distance_matrix = self._create_distance_matrix()
    
    def _create_distance_matrix(self) -> np.ndarray:
        """Create symmetric distance matrix from OD matrix."""
        # Initialize symmetric matrix
        dist_matrix = np.zeros((self.n_locations, self.n_locations))
        
        # Create mapping from location_id to matrix index
        id_to_idx = {loc_id: idx for idx, loc_id in enumerate(self.location_ids)}
        
        # Fill distance matrix
        for row in self.od_matrix.iter_rows():
            origin_id, dest_id, drive_time = row[0], row[1], row[5]  # drive_time_minutes
            
            origin_idx = id_to_idx[origin_id]
            dest_idx = id_to_idx[dest_id]
            
            dist_matrix[origin_idx, dest_idx] = drive_time
            dist_matrix[dest_idx, origin_idx] = drive_time  # Symmetric
        
        return dist_matrix
    
    def cluster_locations(self, n_clusters: int, max_locations_per_cluster: int = 7) -> Dict[int, List[int]]:
        """
        Cluster locations into days using hierarchical clustering.
        
        Args:
            n_clusters: Number of clusters (available secondary days)
            max_locations_per_cluster: Maximum locations per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of location_ids
        """
        # Handle empty case
        if self.n_locations == 0:
            return {}
        
        # Handle single location case
        if self.n_locations == 1:
            return {1: self.location_ids.copy()}
        
        # Convert distance matrix to condensed form for scipy
        condensed_distances = squareform(self.distance_matrix)
        
        # Perform hierarchical clustering with average linkage
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Get cluster assignments
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group locations by cluster
        cluster_assignments = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_assignments:
                cluster_assignments[cluster_id] = []
            cluster_assignments[cluster_id].append(self.location_ids[idx])
        
        # Handle constraint violations (clusters too large)
        cluster_assignments = self._enforce_cluster_size_constraints(
            cluster_assignments, max_locations_per_cluster
        )
        
        return cluster_assignments
    
    def _enforce_cluster_size_constraints(
        self, 
        clusters: Dict[int, List[int]], 
        max_size: int
    ) -> Dict[int, List[int]]:
        """Split clusters that exceed size limit."""
        adjusted_clusters = {}
        cluster_counter = 1
        
        for cluster_id, locations in clusters.items():
            if len(locations) <= max_size:
                # Cluster is within size limit
                adjusted_clusters[cluster_counter] = locations
                cluster_counter += 1
            else:
                # Split oversized cluster
                n_splits = (len(locations) + max_size - 1) // max_size  # Ceiling division
                
                for i in range(n_splits):
                    start_idx = i * max_size
                    end_idx = min((i + 1) * max_size, len(locations))
                    adjusted_clusters[cluster_counter] = locations[start_idx:end_idx]
                    cluster_counter += 1
        
        return adjusted_clusters
    
    def calculate_cluster_quality(self, clusters: Dict[int, List[int]]) -> Dict[str, float]:
        """Calculate quality metrics for clustering solution."""
        total_intra_cluster_distance = 0
        total_pairs = 0
        cluster_sizes = []
        
        for cluster_id, locations in clusters.items():
            cluster_sizes.append(len(locations))
            
            # Calculate average intra-cluster distance
            for i, loc1 in enumerate(locations):
                for j, loc2 in enumerate(locations):
                    if i < j:  # Avoid double counting
                        idx1 = self.location_ids.index(loc1)
                        idx2 = self.location_ids.index(loc2)
                        distance = self.distance_matrix[idx1, idx2]
                        total_intra_cluster_distance += distance
                        total_pairs += 1
        
        avg_intra_cluster_distance = total_intra_cluster_distance / max(total_pairs, 1)
        
        return {
            'avg_intra_cluster_distance': avg_intra_cluster_distance,
            'cluster_size_std': np.std(cluster_sizes),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'n_clusters': len(clusters)
        }
    
    def optimize_with_swaps(
        self, 
        clusters: Dict[int, List[int]], 
        max_iterations: int = 100
    ) -> Dict[int, List[int]]:
        """
        Improve clustering by swapping locations between clusters.
        
        Args:
            clusters: Initial cluster assignments
            max_iterations: Maximum number of swap attempts
            
        Returns:
            Improved cluster assignments
        """
        current_clusters = {k: v.copy() for k, v in clusters.items()}
        best_quality = self.calculate_cluster_quality(current_clusters)['avg_intra_cluster_distance']
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try swapping each location to each other cluster
            for cluster_id, locations in list(current_clusters.items()):
                for location in locations.copy():
                    for other_cluster_id in current_clusters:
                        if other_cluster_id == cluster_id:
                            continue
                        
                        # Skip if target cluster would exceed size limit
                        if len(current_clusters[other_cluster_id]) >= 7:
                            continue
                        
                        # Try the swap
                        current_clusters[cluster_id].remove(location)
                        current_clusters[other_cluster_id].append(location)
                        
                        # Check if improvement
                        new_quality = self.calculate_cluster_quality(current_clusters)['avg_intra_cluster_distance']
                        
                        if new_quality < best_quality:
                            # Keep the swap
                            best_quality = new_quality
                            improved = True
                            break
                        else:
                            # Revert the swap
                            current_clusters[other_cluster_id].remove(location)
                            current_clusters[cluster_id].append(location)
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break  # No more improvements possible
        
        return current_clusters
    
    def assign_days(
        self, 
        available_secondary_days: int,
        max_locations_per_day: int = 7,
        use_swap_optimization: bool = True
    ) -> Dict[int, List[int]]:
        """
        Main method to assign locations to days.
        
        Args:
            available_secondary_days: Number of days available for secondary locations
            max_locations_per_day: Maximum locations per day
            use_swap_optimization: Whether to apply swap optimization
            
        Returns:
            Dictionary mapping day_id to list of location_ids
        """
        # Initial clustering
        clusters = self.cluster_locations(available_secondary_days, max_locations_per_day)
        
        # Optional swap optimization
        if use_swap_optimization:
            clusters = self.optimize_with_swaps(clusters)
        
        return clusters


def load_locations_data(locations_path: str = "data/subway_locations.json") -> List[Dict]:
    """Load and return secondary locations data."""
    import json
    
    with open(locations_path, 'r') as f:
        data = json.load(f)
    
    return [
        loc for loc in data['subway_locations_san_francisco'] 
        if loc['class'] == 'secondary'
    ]


if __name__ == "__main__":
    # Example usage
    optimizer = DayAssignmentOptimizer()
    
    # Assign to 3 days (assuming 5 work days - 2 primary days = 3 secondary days)
    day_assignments = optimizer.assign_days(available_secondary_days=3)
    
    # Display results
    locations_data = load_locations_data()
    id_to_name = {loc['id']: loc['name'] for loc in locations_data}
    
    print("Day Assignments:")
    print("================")
    for day_id, location_ids in day_assignments.items():
        print(f"\nDay {day_id} ({len(location_ids)} locations):")
        for loc_id in location_ids:
            print(f"  - {id_to_name[loc_id]}")
    
    # Show quality metrics
    quality = optimizer.calculate_cluster_quality(day_assignments)
    print(f"\nCluster Quality Metrics:")
    for metric, value in quality.items():
        print(f"  {metric}: {value:.2f}")