# test_pipeline.py
import pandas as pd
from src.config import RAW_DATA_PATH, NUMERIC, CATEGORICAL, TEXT_COL, DATE_COL
from src.data_prep import build_preprocess_pipeline

def main():
    print("📂 Loading dataset from:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    print("Shape before preprocessing:", df.shape)

    # Build and apply preprocessing pipeline
    pipe = build_preprocess_pipeline(NUMERIC, CATEGORICAL, TEXT_COL, DATE_COL)
    X_transformed = pipe.fit_transform(df)

    print("Shape after preprocessing:", X_transformed.shape)

if __name__ == "__main__":
    main()
