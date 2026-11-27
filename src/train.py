"""Main script: load the data, train each model and show its predictions.

Run from the project folder:  python src/train.py
"""
import pandas as pd

from data import clean_data, load_data, split_data
from models import get_models
from features import build_preprocessor, build_pipeline


def main():
    df = clean_data(load_data())
    print(df.head())

    X_train, X_test, y_train, y_test = split_data(df)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    for name, model in get_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)          # Learn
        predictions = pipeline.predict(X_test)  # Predict
        print(f"\n{name} predictions:")
        print(pd.Series(predictions).value_counts())


# Runs main() only when this file is run directly, not when it is imported
if __name__ == "__main__":
    main()
