import numpy as np
import csv
from collections import Counter
from math import log2
from random import Random
from typing import List, Dict, Tuple

class DataPreprocessor:
    def __init__(self):
        self.categorical_mappings = {}
        self.numerical_columns = [
            'Age', 'CGPA', 'Work/Study Hours', 'Sleep Duration'
        ]
        self.categorical_columns = [
            'Academic Pressure', 'Work Pressure', 'Study Satisfaction',
            'Job Satisfaction', 'Financial Stress', 'Gender',
            'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness', 'Dietary Habits'
        ]
    
    def encode_categorical(self, column_name: str, values: List[str]) -> List[int]:
        # Create mapping if it doesn't exist
        if column_name not in self.categorical_mappings:
            unique_values = list(set([v for v in values if v != '']))
            self.categorical_mappings[column_name] = {
                val: idx for idx, val in enumerate(unique_values)
            }
        
        # Encode values using mapping
        mapping = self.categorical_mappings[column_name]
        return [mapping.get(val, -1) if val != '' else -1 for val in values]
    
    def convert_numerical(self, values: List[str]) -> List[float]:
        # Convert to float, handle empty strings
        return [float(val) if val != '' else -1 for val in values]

def load_csv(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess CSV file with specific column handling."""
    preprocessor = DataPreprocessor()
    
    # First pass: read all data
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Prepare data structures
    X_processed = []
    y = []
    
    # Process each column
    columns_data = {col: [row[col] for row in rows] for col in rows[0].keys()}
    
    # Process features
    X_columns = []
    
    # Process numerical columns
    for col in preprocessor.numerical_columns:
        X_columns.append(preprocessor.convert_numerical(columns_data[col]))
    
    # Process categorical columns
    for col in preprocessor.categorical_columns:
        X_columns.append(preprocessor.encode_categorical(col, columns_data[col]))
    
    # Convert to numpy array and transpose
    X = np.array(X_columns).T
    
    # Process target variable (Depression)
    y = preprocessor.encode_categorical('Depression', columns_data['Depression'])
    
    # Remove rows with missing values
    valid_rows = ~np.any(X == -1, axis=1)
    X = X[valid_rows]
    y = np.array(y)[valid_rows]
    
    return X, y

class Node:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.predicted_class = None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.random = Random(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(set(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0) -> Node:
        n_samples, n_features = X.shape
        node = Node()

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(set(y)) == 1:
            node.is_leaf = True
            node.predicted_class = self._most_common_label(y)
            return node

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No valid split found
            node.is_leaf = True
            node.predicted_class = self._most_common_label(y)
            return node

        # Create child nodes
        node.feature_index = best_feature
        node.threshold = best_threshold

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Sample subset of features for random forest
        feature_indices = self.random.sample(range(n_features), 
                                          max(1, int(np.sqrt(n_features))))

        for feature_index in feature_indices:
            thresholds = sorted(set(X[:, feature_index]))
            
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                gain = self._information_gain(X[:, feature_index], y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, 
                         threshold: float) -> float:
        parent_entropy = self._entropy(y)

        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if not any(left_mask) or not any(right_mask):
            return 0

        n = len(y)
        n_l, n_r = sum(left_mask), sum(right_mask)
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_l * e_l + n_r * e_r) / n

        return parent_entropy - child_entropy

    def _entropy(self, y: np.ndarray) -> float:
        hist = Counter(y)
        ps = [count / len(y) for count in hist.values()]
        return -sum(p * log2(p) for p in ps)

    def _most_common_label(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray):
        node = self.root
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.randint(0, len(X), size=len(X))
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(pred).most_common(1)[0][0] 
                        for pred in predictions.T])

def train_test_split(X: np.ndarray, y: np.ndarray, 
                     test_size: float, random_state: int = None) -> Tuple:
    """Split arrays into random train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix."""
    n_classes = max(max(y_true), max(y_pred)) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate classification report."""
    labels = sorted(set(y_true))
    report = []
    
    # Calculate metrics for each class
    for label in labels:
        true_pos = sum((y_true == label) & (y_pred == label))
        false_pos = sum((y_true != label) & (y_pred == label))
        false_neg = sum((y_true == label) & (y_pred != label))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report.append(f"Class {label}:")
        report.append(f"    Precision: {precision:.2f}")
        report.append(f"    Recall: {recall:.2f}")
        report.append(f"    F1-score: {f1:.2f}")
    
    return "\n".join(report)

# Example usage:
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_csv('preprocessed_student.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                       random_state=42)
    
    # Create and train the model
    model = RandomForest(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


'''
Class 0 appears to be "Not Depressed"
Class 1 appears to be "Depressed"

For Class 0 (Not Depressed):
Precision (0.84): When the model predicts someone is not depressed, it's correct 84% of the time
Recall (0.89): Of all people who are actually not depressed, the model correctly identifies 89% of them
F1-score (0.86): A balanced average of precision and recall

For Class 1 (Depressed):
Precision (0.83): When the model predicts someone is depressed, it's correct 83% of the time
Recall (0.76): Of all people who are actually depressed, the model correctly identifies 76% of them
F1-score (0.80): A balanced average of precision and recall

Key observations:
The model is slightly better at identifying non-depressed cases (Class 0) than depressed cases (Class 1)
For depressed cases, the model has more false negatives (missing some cases of depression) than false positives (incorrectly labeling someone as depressed)
Overall, these are fairly balanced scores, suggesting the model performs reasonably well for both classes
'''

'''
50 estimator 
Accuracy: 83.74%

Classification Report:
Class 0:
    Precision: 0.84
    Recall: 0.89
    F1-score: 0.86
Class 1:
    Precision: 0.83
    Recall: 0.76
    F1-score: 0.80

100 estimator
Accuracy: 84.26%

Classification Report:
Class 0:
    Precision: 0.84
    Recall: 0.78
    F1-score: 0.80
Class 1:
    Precision: 0.85
    Recall: 0.89
    F1-score: 0.87
'''