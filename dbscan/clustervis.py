import numpy as np
import matplotlib.pyplot as plt
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

def perform_pca(X, n_components=2):
    """Manually perform PCA without using sklearn."""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    X_pca = np.dot(X_centered, selected_eigenvectors)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance_ratio

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
        
        return self.labels

def visualize_clusters_pca(data, labels):
    """Visualize clusters after PCA transformation."""
    # Perform PCA
    data_pca, explained_variance_ratio = perform_pca(data)
    
    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    
    # Generate colors for clusters
    colors = ['black'] + [plt.cm.jet(i/float(len(unique_clusters)-1)) 
                         for i in range(len(unique_clusters)-1)]
    
    # Plot each cluster
    for cluster_id, color in zip(unique_clusters, colors):
        mask = labels == cluster_id
        label = 'Noise' if cluster_id == 0 else f'Cluster {cluster_id}'
        plt.scatter(data_pca[mask, 0], data_pca[mask, 1], 
                   c=[color], label=label, alpha=0.6)

    plt.title('DBSCAN Clustering Results (PCA)')
    plt.xlabel(f'First Principal Component\n'
              f'(Explained Variance Ratio: {explained_variance_ratio[0]:.2%})')
    plt.ylabel(f'Second Principal Component\n'
              f'(Explained Variance Ratio: {explained_variance_ratio[1]:.2%})')
    plt.legend()
    plt.grid(True)
    
    # Add total explained variance in the corner
    total_var = sum(explained_variance_ratio)
    plt.text(0.02, 0.98, f'Total Explained Variance: {total_var:.2%}',
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read data
    data = read_data("../preprocessed_student.csv")
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    labels = dbscan.fit(data)
    
    # Visualize results with PCA
    visualize_clusters_pca(data, labels)