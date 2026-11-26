"""Create the models to compare."""
from sklearn.dummy import DummyClassifier


def get_models():
    """Return the models to train, as {name: model}."""
    return {
        "dummy": DummyClassifier(),  # Baseline: always predicts the most common answer
        # TODO: logistic regression, decision tree, random forest
    }
