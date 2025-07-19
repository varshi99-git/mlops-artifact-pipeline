# src/train.py.
import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    config = load_config(r"C:/Users/saiva/mlops-artifact-pipeline/src/config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)

    with open("model_train.pkl", "wb") as f:
        pickle.dump(model, f)
