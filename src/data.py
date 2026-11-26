"""Load, clean and split the loan data."""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_data():
    """Read the raw CSV into a DataFrame."""
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    """Fix problems in the raw data. The steps are listed in docs/project_brief.md."""
    # TODO: rename Attribute1 to checking_balance
    # TODO: merge car0 into car in the purpose column
    # TODO: drop age
    return df


def split_data(df):
    """Split into inputs (X) and target (y), then into train and test sets."""
    X = df.drop(TARGET, axis=1)  # The questions (inputs)
    y = df[TARGET]               # The answers (target)

    # Returns X_train, X_test, y_train, y_test
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
