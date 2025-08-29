# src/config.py
from dataclasses import dataclass

# Path to your dataset
RAW_DATA_PATH = "data/raw/insurance.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target column (the one we want to predict)
TARGET = "Premium Amount"

# Special columns
DATE_COL = "Policy Start Date"
TEXT_COL = "Customer Feedback"

# Categorical columns in dataset
CATEGORICAL = [
    "Gender", "Marital Status", "Education Level", "Occupation",
    "Location", "Policy Type", "Smoking Status", "Exercise Frequency",
    "Property Type"
]

# Numeric columns in dataset
NUMERIC = [
    "Age", "Annual Income", "Number of Dependents", "Health Score",
    "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"
]
