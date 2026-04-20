from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def train_model():
    data = load_iris()
    X, y = data.data, data.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model
