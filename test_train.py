import unittest
from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

CONFIG_PATH = "./src/config/config.json"

# ----------------------------
# UNIT TESTS
# ----------------------------
class UnitTestConfig(unittest.TestCase):

    def test_config_loading_types(self):
        config = load_config(CONFIG_PATH)
        self.assertIsInstance(config["C"], float, "C should be a float")
        self.assertIsInstance(config["solver"], str, "solver should be a string")
        self.assertIsInstance(config["max_iter"], int, "max_iter should be an integer")


class UnitTestModel(unittest.TestCase):

    def test_model_training_output(self):
        digits = load_digits()
        X, y = digits.data, digits.target
        config = load_config(CONFIG_PATH)
        model = train_model(X, y, config)
        self.assertIsInstance(model, LogisticRegression, "Model should be LogisticRegression")
        self.assertTrue(hasattr(model, "coef_"), "Model should have coefficients after training")


# ----------------------------
# INTEGRATION TESTS
# ----------------------------
class IntegrationTestModelPipeline(unittest.TestCase):

    def test_model_accuracy_above_threshold(self):
        digits = load_digits()
        X, y = digits.data, digits.target
        config = load_config(CONFIG_PATH)
        model = train_model(X, y, config)
        score = model.score(X, y)
        self.assertGreater(score, 0.8, "Model accuracy should be above 0.8")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    unittest.main()
