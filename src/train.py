# train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import randint, uniform
import xgboost as xgb
from data_prep import FeatureEngineeringTransformer
# ==========================
# 1. Feature Engineering
# ==========================
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Policy Age
        if "Policy Start Date" in X.columns:
            X["Policy Start Date"] = pd.to_datetime(X["Policy Start Date"], errors="coerce")
            X["Policy_Age"] = (pd.to_datetime("today") - X["Policy Start Date"]).dt.days // 365
            X.drop(columns=["Policy Start Date"], inplace=True)
        
        # Feedback Sentiment
        if "Customer Feedback" in X.columns:
            def sentiment_score(text):
                if pd.isna(text):
                    return 0
                return TextBlob(str(text)).sentiment.polarity
            X["Feedback_Score"] = X["Customer Feedback"].apply(sentiment_score)
            X.drop(columns=["Customer Feedback"], inplace=True)
        
        # Log-transform skewed features
        for col in ["Annual Income", "Credit Score"]:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        
        return X

# ==========================
# 2. Load Data
# ==========================
print("📂 Loading dataset...")
data = pd.read_csv("../data/raw/insurance.csv")
target_column = "Premium Amount"

X = data.drop(target_column, axis=1)
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 3. Build Preprocessor
# ==========================
def build_preprocessor(X, target_column):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, categorical_cols),
        ("num", num_pipeline, numerical_cols)
    ])
    
    return preprocessor

# ==========================
# 4. Build Full Pipeline
# ==========================
fe = FeatureEngineeringTransformer()
preprocessor = build_preprocessor(fe.transform(X_train), target_column)

pipeline = Pipeline([
    ("feature_engineering", fe),
    ("preprocessing", preprocessor),
    ("model", xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])

# ==========================
# 5. Hyperparameter Tuning
# ==========================
param_dist = {
    "model__n_estimators": randint(100, 300),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 6)
}

search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=15, cv=3,
    scoring="r2", verbose=1, random_state=42, n_jobs=-1
)

# ==========================
# 6. Train Model
# ==========================
print("🤖 Training XGBoost pipeline with hyperparameter tuning...")
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_

# ==========================
# 7. Evaluate
# ==========================
y_pred = best_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================
# 8. Save Final Pipeline
# ==========================
joblib.dump(best_pipeline, "../models/final_premium_xgb_pipeline.pkl")
print("\n✅ Final XGBoost pipeline saved to ../models/final_premium_xgb_pipeline.pkl")

print("\n🎉 Training complete!")
