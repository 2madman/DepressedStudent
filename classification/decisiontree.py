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
        y = np.round(y).astype(int)
        self.n_classes = len(set(y))
        self.root = self._grow_tree(X, y)
        
    def _entropy(self, y):
        
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
    y_pred = clf.predict(X_test)
    
    # Calculate and print test set metrics
    test_metrics = calculate_metrics(y_test, y_pred)
    print("\nTest Set Metrics:")
    print(f"True Positives (TP): {test_metrics['TP']}")
    print(f"True Negatives (TN): {test_metrics['TN']}")
    print(f"False Positives (FP): {test_metrics['FP']}")
    print(f"False Negatives (FN): {test_metrics['FN']}")
    print(f"Accuracy: {test_metrics['Accuracy']:.3f}")
    print(f"Precision: {test_metrics['Precision']:.3f}")
    print(f"Recall: {test_metrics['Recall']:.3f}")
    print(f"F1 Score: {test_metrics['F1']:.3f}")
    
    # Perform manual k-fold cross validation
    print("\nPerforming 5-fold Cross Validation:")
    folds = manual_k_fold(X_train, y_train, k=5)
    cv_scores = []
    
    for fold_idx, (X_fold_train, X_fold_val, y_fold_train, y_fold_val) in enumerate(folds, 1):
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X_fold_train, y_fold_train)
        y_fold_pred = clf.predict(X_fold_val)
        
        fold_metrics = calculate_metrics(y_fold_val, y_fold_pred)
        cv_scores.append(fold_metrics['Accuracy'])
        
        print(f"\nFold {fold_idx} Results:")
        print(f"TP: {fold_metrics['TP']}, TN: {fold_metrics['TN']}")
        print(f"FP: {fold_metrics['FP']}, FN: {fold_metrics['FN']}")
        print(f"Accuracy: {fold_metrics['Accuracy']:.3f}")
    
    print(f"\nCross-validation mean accuracy: {np.mean(cv_scores):.3f}")
    print(f"Cross-validation std accuracy: {np.std(cv_scores):.3f}")

if __name__ == "__main__":
    main()