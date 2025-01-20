import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('../preprocessed_student.csv')

# Separate features and target
X = df.drop(['Depression'], axis=1)  # Features
y = df['Depression']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=75,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42
)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate various scores
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

# Print results
print("\nModel Performance Metrics:")
print("-------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

print("\nClassification Report:")
print("--------------------")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print("----------------")
print(confusion_matrix(y_test, y_pred))
