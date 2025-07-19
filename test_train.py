import json
from src.train import load_config, train_model
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def test_config_loading():
    config = load_config("./src/config/config.json")
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("./src/config/config.json")
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("./src/config/config.json")
    model = train_model(X, y, config)
    score = model.score(X, y)
    assert score > 0.8
