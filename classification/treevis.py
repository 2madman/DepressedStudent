import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def manual_train_test_split(X, y, test_size=0.2):
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def manual_k_fold(X, y, k=5):
    n_samples = len(y)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k-1 else n_samples
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        folds.append((X_train, X_val, y_train, y_val))
    
    return folds

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        # Convert y to integer explicitly
        y = np.round(y).astype(int)
        self.n_classes = len(set(y))
        self.root = self._grow_tree(X, y)
        
    def _entropy(self, y):
        # Convert y to integer explicitly
        y = np.round(y).astype(int)
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            # Convert y to integer before getting most common label
            y = np.round(y).astype(int)
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            # Convert y to integer before getting most common label
            y = np.round(y).astype(int)
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def calculate_metrics(y_true, y_pred):
    # Convert both arrays to integer explicitly
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def get_tree_structure(node, feature_names=None, depth=0, pos=0, width=1):
    """Extract the structure of the tree for visualization"""
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(100)]  # Default feature names
        
    info = {
        'depth': depth,
        'pos': pos,
        'width': width,
        'feature': feature_names[node.feature] if node.feature is not None else None,
        'threshold': node.threshold,
        'value': node.value,
        'left': None,
        'right': None
    }
    
    if node.left:
        info['left'] = get_tree_structure(
            node.left,
            feature_names,
            depth + 1,
            pos - width/2,
            width/2
        )
    if node.right:
        info['right'] = get_tree_structure(
            node.right,
            feature_names,
            depth + 1,
            pos + width/2,
            width/2
        )
    
    return info

def draw_node(ax, node_info, parent_pos=None):
    """Draw a node and its connections"""
    x = node_info['pos']
    y = -node_info['depth']
    
    # Draw connection to parent
    if parent_pos is not None:
        ax.plot([parent_pos[0], x], [parent_pos[1], y], 'gray', linewidth=1)
    
    # Create node text
    if node_info['value'] is not None:
        text = f'Class: {node_info["value"]}'
        node_color = 'lightgreen'
    else:
        text = f'{node_info["feature"]}\n<{node_info["threshold"]:.2f}'
        node_color = 'lightblue'
    
    # Draw node
    circle = plt.Circle((x, y), 0.2, color=node_color, alpha=0.3)
    ax.add_patch(circle)
    
    # Add text
    ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # Recursively draw children
    if node_info['left']:
        draw_node(ax, node_info['left'], (x, y))
    if node_info['right']:
        draw_node(ax, node_info['right'], (x, y))

def visualize_tree(tree, feature_names=None, figsize=(12, 8)):
    """Main function to visualize the decision tree"""
    # Get tree structure
    tree_structure = get_tree_structure(tree.root, feature_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw tree
    draw_node(ax, tree_structure)
    
    # Set plot properties
    max_depth = 4  # Assuming max_depth=4 from your code
    ax.set_xlim(-2, 2)
    ax.set_ylim(-max_depth-0.5, 0.5)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title('Decision Tree Visualization')
    plt.tight_layout()
    
    return fig

# Example usage (modify the main() function to include this):
def main():
    # Load data
    data = pd.read_csv('../preprocessed_student.csv')
    
    # Separate features and target
    X = data.drop(['Depression'], axis=1).values
    y = data['Depression'].values
    
    # Convert y to integer type right after loading
    y = np.round(y).astype(int)
    
    # Manual train-test split
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)
    
    # Train and evaluate on test set
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)

    # After training the classifier, add:
    feature_names = data.drop(['Depression'], axis=1).columns.tolist()
    fig = visualize_tree(clf, feature_names)
    plt.show()
    

if __name__ == "__main__":
    main()