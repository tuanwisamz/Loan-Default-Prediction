"""Create the models to compare."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """Return the models to train, as {name: model}."""
    return {
        "dummy": DummyClassifier(),  # Baseline: always predicts the most common answer
        "Logistic_Regression": LogisticRegression(),
        "Decision_Tree": DecisionTreeClassifier(),
        "Random_Forest": RandomForestClassifier()
        
    }
