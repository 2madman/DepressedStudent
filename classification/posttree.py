import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from copy import deepcopy
from decisiontree import DecisionTreeClassifier, manual_train_test_split, calculate_metrics

class DecisionTreePostProcessor:
    def __init__(self, tree, feature_names=None):
        """
        Initialize postprocessor with a trained DecisionTreeClassifier
        
        Parameters:
        tree: DecisionTreeClassifier
            A trained decision tree instance
        feature_names: list
            List of feature names for better visualization
        """
        self.tree = tree
        self.feature_names = feature_names

    def analyze_feature_importance(self, X, y):
        """Calculate and visualize feature importance"""
        feature_importance = np.zeros(X.shape[1])
        total_samples = len(y)
        
        def traverse_tree(node, n_samples):
            if node.value is not None:  # Leaf node
                return
                
            feature = node.feature
            threshold = node.threshold
            
            gain = self.tree._information_gain(y, X[:, feature], threshold)
            feature_importance[feature] += gain * (n_samples / total_samples)
            
            left_idxs = X[:, feature] < threshold
            right_idxs = ~left_idxs
            n_left = np.sum(left_idxs)
            n_right = np.sum(right_idxs)
            
            traverse_tree(node.left, n_left)
            traverse_tree(node.right, n_right)
        
        traverse_tree(self.tree.root, total_samples)
        
        # Normalize feature importances
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create feature importance DataFrame
        feature_df = pd.DataFrame({
            'Feature': self.feature_names if self.feature_names else [f'Feature {i}' for i in range(len(feature_importance))],
            'Importance': feature_importance
        })
        feature_df = feature_df.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_df, x='Importance', y='Feature')
        plt.title('Feature Importance in Depression Prediction', pad=20)
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_df

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Not Depressed', 'Depressed']):
        """Plot confusion matrix with class names"""
        n_classes = self.tree.n_classes
        cm = np.zeros((n_classes, n_classes))
        
        # Convert predictions and true values to integers
        y_true = np.round(y_true).astype(int)
        y_pred = np.round(y_pred).astype(int)
        
        for i in range(len(y_true)):
            cm[y_true[i]][y_pred[i]] += 1
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix', pad=20)
        plt.ylabel('')
        plt.xlabel('')
        plt.tight_layout()
        plt.show()
        
        return cm

    def depth_performance_analysis(self, X_train, y_train, X_val, y_val, max_depths=range(1, 11)):
        """Analyze model performance across different tree depths"""
        train_scores = []
        val_scores = []
        
        for depth in max_depths:
            temp_tree = DecisionTreeClassifier(max_depth=depth)
            temp_tree.fit(X_train, y_train)
            
            train_pred = temp_tree.predict(X_train)
            val_pred = temp_tree.predict(X_val)
            
            train_acc = np.mean(train_pred == y_train)
            val_acc = np.mean(val_pred == y_val)
            
            train_scores.append(train_acc)
            val_scores.append(val_acc)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(max_depths, train_scores, 'o-', label='Training Accuracy')
        plt.plot(max_depths, val_scores, 'o-', label='Validation Accuracy')
        plt.xlabel('Maximum Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Model Performance vs Tree Depth', pad=20)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'Max Depth': max_depths,
            'Training Accuracy': train_scores,
            'Validation Accuracy': val_scores
        })

def main():
    # Load and prepare data
    data = pd.read_csv('../preprocessed_student.csv')
    feature_names = [col for col in data.columns if col != 'Depression']
    X = data.drop(['Depression'], axis=1).values
    y = data['Depression'].values.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)
    
    # Train oriTnal tree
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)
    
    # Initialize postprocessor with feature names
    post_processor = DecisionTreePostProcessor(clf, feature_names)
    
    print("\n=== Decision Tree Analysis Report ===")
    
    # 1. Feature Importance Analysis
    print("\n1. Feature Importance Analysis")
    print("-" * 40)
    feature_importance_df = post_processor.analyze_feature_importance(X_train, y_train)
    print("\nTop 5 Most Important Features:")
    print(feature_importance_df.head().to_string(index=False))
    
    # 2. Model Performance Metrics
    print("\n2. Model Performance Metrics")
    print("-" * 40)
    y_pred = clf.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\nAccuracy: {metrics['Accuracy']:.3f}")
    print(f"Precision: {metrics['Precision']:.3f}")
    print(f"Recall: {metrics['Recall']:.3f}")
    print(f"F1 Score: {metrics['F1']:.3f}")
    
    # 3. Confusion Matrix
    print("\n3. Confusion Matrix Visualization")
    print("-" * 40)
    print("Plotting confusion matrix...")
    cm = post_processor.plot_confusion_matrix(y_test, y_pred)
    
    # 4. Depth Analysis
    print("\n4. Tree Depth Analysis")
    print("-" * 40)
    depth_results = post_processor.depth_performance_analysis(
        X_train, y_train, X_test, y_test
    )
    print("\nAccuracy at different tree depths:")
    print(depth_results.round(3).to_string(index=False))

if __name__ == "__main__":
    main()