import numpy as np
from dbscan import DBSCAN, read_data

class ClusterAnalysis:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.feature_names = [
            "Age", "Academic Pressure", "CGPA", 
            "Study Satisfaction", "Work/Study Hours",
            "Financial Stress", "Gender", "Suicidal Thoughts",
            "Family History", "Depression", "Sleep Duration", "Dietary Habits"
        ]

    def analyze_clusters(self):
        """Perform detailed analysis of clusters."""
        cluster_stats = {}
        overall_mean = np.mean(self.data, axis=0)
        
        # Analyze each cluster
        for label in set(self.labels):
            if label != 0:  # Skip noise points
                mask = self.labels == label
                cluster_data = self.data[mask]
                
                # Calculate statistics for each feature
                feature_stats = []
                for i, feature in enumerate(self.feature_names):
                    cluster_values = cluster_data[:, i]
                    other_values = self.data[~mask, i]
                    
                    stats = {
                        'feature': feature,
                        'cluster_mean': np.mean(cluster_values),
                        'other_mean': np.mean(other_values),
                        'difference': np.mean(cluster_values) - np.mean(other_values),
                        'cluster_std': np.std(cluster_values),
                        'other_std': np.std(other_values)
                    }
                    
                    # Calculate percent difference
                    if stats['other_mean'] != 0:
                        stats['percent_diff'] = (stats['difference'] / stats['other_mean']) * 100
                    else:
                        stats['percent_diff'] = 0
                        
                    feature_stats.append(stats)
                
                # Sort features by absolute percent difference
                significant_features = sorted(
                    feature_stats, 
                    key=lambda x: abs(x['percent_diff']), 
                    reverse=True
                )
                
                cluster_stats[label] = {
                    'size': len(cluster_data),
                    'features': significant_features
                }

        self._print_analysis(cluster_stats)
        return cluster_stats

    def _print_analysis(self, cluster_stats):
        """Print formatted analysis results."""
        print("\n=== Detailed Cluster Analysis ===")
        for label, stats in sorted(cluster_stats.items()):
            print(f"\nCluster {label} ({stats['size']} points)")
            print("Characteristics (compared to other points):")
            
            # Print top features for this cluster
            for feat in stats['features'][:5]:  # Show top 5 features
                diff = feat['percent_diff']
                if abs(diff) > 1:  # Show only if difference is more than 1%
                    direction = "higher" if diff > 0 else "lower"
                    print(f"\n- {feat['feature']}:")
                    print(f"  {abs(diff):.1f}% {direction} than other points")
                    print(f"  Cluster mean: {feat['cluster_mean']:.3f}")
                    print(f"  Others mean: {feat['other_mean']:.3f}")
                    print(f"  Cluster std: {feat['cluster_std']:.3f}")
                    print(f"  Others std: {feat['other_std']:.3f}")

def main():
    data = read_data("../preprocessed_student.csv")
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    labels = dbscan.fit(data)
    
    # Perform analysis
    analyzer = ClusterAnalysis(data, labels)
    cluster_stats = analyzer.analyze_clusters()

if __name__ == "__main__":
    main()