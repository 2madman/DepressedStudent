import numpy as np
from collections import defaultdict
import time

def read_data(filename):
    """Read and parse the CSV data file."""
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            try:
                values = line.strip().split(',')
                float_values = [float(val) if val.strip() != '' else 0.0 for val in values]
                if float_values:
                    data.append(float_values)
            except Exception as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
    return np.array(data)

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _find_neighbors(self, data, point_idx):
        """Find neighbors using vectorized computation."""
        distances = np.sqrt(np.sum((data - data[point_idx]) ** 2, axis=1))
        neighbors = np.where(distances <= self.eps)[0]
        return [idx for idx in neighbors if idx != point_idx]

    def fit(self, data):
        n_samples = len(data)
        self.labels = np.full(n_samples, -1)
        
        # Initialize progress tracking
        start_time = time.time()
        last_update = start_time
        update_interval = 2
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Update progress
            current_time = time.time()
            if current_time - last_update >= update_interval:
                progress = (point_idx + 1) / n_samples * 100
                elapsed = current_time - start_time
                print(f"Progress: {progress:.1f}% | Points processed: {point_idx + 1}/{n_samples} | "
                      f"Elapsed time: {elapsed:.1f}s")
                last_update = current_time
            
            if self.labels[point_idx] != -1:
                continue
            
            neighbors = self._find_neighbors(data, point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = 0  # Noise point
                continue
            
            # Start new cluster
            cluster_id += 1
            self.labels[point_idx] = cluster_id
            
            # Process neighbors
            seed_set = set(neighbors)
            to_process = list(neighbors)
            
            while to_process:
                current_point = to_process.pop(0)
                if self.labels[current_point] == 0:
                    self.labels[current_point] = cluster_id
                elif self.labels[current_point] != -1:
                    continue
                
                self.labels[current_point] = cluster_id
                current_neighbors = self._find_neighbors(data, current_point)
                
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in seed_set:
                            seed_set.add(neighbor)
                            to_process.append(neighbor)
        
        # Print final statistics
        elapsed = time.time() - start_time
        n_clusters = len(set(self.labels)) - (1 if 0 in self.labels else 0)
        n_noise = list(self.labels).count(0)
        
        print("\nDBSCAN clustering completed:")
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Number of clusters found: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        cluster_sizes = self.get_cluster_sizes()
        for cluster_id, size in sorted(cluster_sizes.items()):
            if cluster_id == 0:
                print(f"Noise points: {size}")
            else:
                print(f"Cluster {cluster_id}: {size} points")        
        return self.labels
    
    def get_cluster_sizes(self):
        """Return a dictionary of cluster sizes."""
        if self.labels is None:
            return {}
        sizes = defaultdict(int)
        for label in self.labels:
            sizes[label] += 1
        return dict(sizes)

if __name__ == "__main__":
    # Read data (no scaling since data is already normalized)
    data = read_data("preprocessed_student.csv")
    
    # Run DBSCAN with adjusted parameters for normalized data
    dbscan = DBSCAN(eps=0.45, min_samples=10)
    labels = dbscan.fit(data)