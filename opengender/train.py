import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification

from opengender.dame_sexmachine import features_list, csv2gender_list
from opengender.paths import ALL_PATH, DATA_DIR


def train_svc():
    X = features_list(path=ALL_PATH)
    y = csv2gender_list(path=ALL_PATH)

    model = SVC()
    model.fit(X, y)

    pickle.dump(model, open(DATA_DIR / "svc.pkl", "wb"))


def train_forest():
    X = features_list(path=ALL_PATH)
    y = csv2gender_list(path=ALL_PATH)

    X, y = make_classification(
        n_samples=7000,
        n_features=33,
        n_informative=33,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )

    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model.fit(X, y)

    with open(DATA_DIR / "forest.pkl", "wb") as fh:
        pickle.dump(model, fh)
