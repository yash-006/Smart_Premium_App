# data_prep.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from textblob import TextBlob
from sklearn.model_selection import train_test_split

# ==========================
# 1. Feature Engineering
# ==========================
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering:
    - Policy Start Date → Policy Age
    - Customer Feedback → Sentiment Score
    - Log-transform skewed numeric features
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Convert Policy Start Date → Policy Age
        if "Policy Start Date" in X.columns:
            X["Policy Start Date"] = pd.to_datetime(X["Policy Start Date"], errors="coerce")
            X["Policy_Age"] = (pd.to_datetime("today") - X["Policy Start Date"]).dt.days // 365
            X.drop(columns=["Policy Start Date"], inplace=True)
        
        # 2. Convert Customer Feedback → Sentiment Score
        if "Customer Feedback" in X.columns:
            def sentiment_score(text):
                if pd.isna(text):
                    return 0
                return TextBlob(str(text)).sentiment.polarity
            X["Feedback_Score"] = X["Customer Feedback"].apply(sentiment_score)
            X.drop(columns=["Customer Feedback"], inplace=True)
        
        # 3. Log-transform skewed features
        for col in ["Annual Income", "Credit Score", "Premium Amount"]:
            if col in X.columns:
                X[col] = np.log1p(X[col])  # log(1+x) to avoid -inf
        
        return X

# ==========================
# 2. Build Preprocessor
# ==========================
def build_preprocessor(X, target_column="Premium Amount"):
    """
    Build ColumnTransformer for preprocessing:
    - Numeric: median imputation + robust scaling
    - Categorical: most frequent imputation + one-hot encoding
    """
    # Identify categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    # Remove target if included
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    
    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numerical_cols)
        ]
    )
    
    return preprocessor

# ==========================
# 3. Load & Split Data
# ==========================
def load_data(path):
    """
    Load dataset from CSV
    """
    df = pd.read_csv(path)
    print(f"📂 Loaded dataset with shape: {df.shape}")
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Split data: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test
