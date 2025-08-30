import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin

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
        for col in ["Annual Income", "Credit Score", "Premium Amount"]:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X
