"""Turn text columns into numbers (one-hot encoding) and scale numeric columns, inside a scikit-learn Pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns left after clean_data() (age is dropped): 6 numeric + 9 text = 15
NUMERIC_COLUMNS = [
    "months_loan_duration",
    "amount",
    "percent_of_income",
    "years_at_residence",
    "existing_loans_count",
    "dependents",
]
CATEGORICAL_COLUMNS = [
    "checking_balance",
    "credit_history",
    "purpose",
    "savings_balance",
    "employment_duration",
    "other_credit",
    "housing",
    "job",
    "phone",
]


def build_preprocessor():
    """Return a ColumnTransformer that turns every column of X into numbers."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
    ])


def build_pipeline(model):
    """Join the preprocessor and a model into one Pipeline: preprocess first, then the model."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model)
    ])
