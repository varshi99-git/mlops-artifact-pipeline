import pickle
from sklearn.datasets import load_digits

with open("model_train.pkl", "rb") as f:
    model = pickle.load(f)

digits = load_digits()
X = digits.data
preds = model.predict(X)

print("Sample Predictions:", preds[:10])
