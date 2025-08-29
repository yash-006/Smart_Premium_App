# src/modeling.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # if xgboost not installed

def get_models():
    """Return dictionary of models for training."""
    models = {
        "linreg": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=42
        )
    }
    if XGBRegressor is not None:
        models["xgb"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.06,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_jobs=-1,
            tree_method="hist"
        )
    return models

# hyperparameter search spaces for Random Forest & XGBoost
SEARCH_SPACES = {
    "rf": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_split": [2, 5, 10]
    },
    "xgb": {
        "model__n_estimators": [400, 600, 900],
        "model__max_depth": [6, 8, 10],
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0]
    }
}
