import pandas as pd
import numpy as np

def normalize_numeric_column(column):
    min_val = column.min()
    max_val = column.max()
    if max_val == min_val:
        return np.zeros(len(column))
    return (column - min_val) / (max_val - min_val)

def encode_binary_column(column):
    unique_values = column.unique()
    return np.where(column == unique_values[0], 0, 1)

def encode_sleep_duration(column):
    sleep_map = {
        'Less than 5 hours': 0.2,
        '5-6 hours': 0.5,
        '7-8 hours': 0.75,
        'More than 8 hours': 1
    }
    return column.map(sleep_map)

def encode_dietary_habits(column):
    diet_map = {
        'Unhealthy': 0.5,
        'Moderate': 0.75,
        'Healthy': 1
    }
    return column.map(diet_map)

def preprocess_student_data(df):
    # Create a copy of the dataframe
    df = df.copy()
    
    # Drop unnecessary columns
    columns_to_drop = ['id', 'City',]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Initialize list to store processed features and their names
    processed_features = []
    feature_names = []
    
    # Process numeric columns
    numeric_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                      'Study Satisfaction', 'Job Satisfaction', 
                      'Work/Study Hours', 'Financial Stress']
    
    for col in numeric_columns:
        normalized = normalize_numeric_column(df[col])
        processed_features.append(normalized)
        feature_names.append(col)
    
    # Process binary columns
    binary_columns = ['Gender', 'Have you ever had suicidal thoughts ?',
                     'Family History of Mental Illness', 'Depression']
    
    for col in binary_columns:
        encoded = encode_binary_column(df[col])
        processed_features.append(encoded)
        feature_names.append(col)
    
    # Process Sleep Duration and Dietary Habits as ordinal variables
    sleep_encoded = encode_sleep_duration(df['Sleep Duration'])
    processed_features.append(sleep_encoded)
    feature_names.append('Sleep Duration')
    
    diet_encoded = encode_dietary_habits(df['Dietary Habits'])
    processed_features.append(diet_encoded)
    feature_names.append('Dietary Habits')
    
    # Combine all processed features into a single matrix
    preprocessed_matrix = np.column_stack(processed_features)
    
    # Convert to DataFrame with feature names
    preprocessed_df = pd.DataFrame(preprocessed_matrix, columns=feature_names)
    
    return preprocessed_df

# Read the original data
df = pd.read_csv('student.csv')

# Preprocess the data
preprocessed_df = preprocess_student_data(df)

# Save the preprocessed data to a new CSV file
preprocessed_df.to_csv('preprocessed_student.csv', index=False)

print("Preprocessing completed. Data saved to 'preprocessed_student.csv'")
print(f"Shape of preprocessed data: {preprocessed_df.shape}")
print("\nFirst few columns of preprocessed data:")
print(preprocessed_df.head())