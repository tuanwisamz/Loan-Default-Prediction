"""Main script: load the data, train each model and show its predictions.

Run from the project folder:  python src/train.py
"""
import pandas as pd

from data import clean_data, load_data, split_data
from models import get_models


def main():
    df = clean_data(load_data())
    print(df.head())

    X_train, X_test, y_train, y_test = split_data(df)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    for name, model in get_models().items():
        model.fit(X_train, y_train)          # Learn
        predictions = model.predict(X_test)  # Predict
        print(f"\n{name} predictions:")
        print(pd.Series(predictions).value_counts())


# Runs main() only when this file is run directly, not when it is imported
if __name__ == "__main__":
    main()
