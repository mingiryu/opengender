import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from opengender.paths import ALL_PATH, DATA_DIR


def features_int(name):
    # features method created to check the scikit classifiers
    features = {}
    features["first_letter"] = ord(name[0].lower())
    features["last_letter"] = ord(name[-1].lower())

    for letter in "abcdefghijklmnopqrstuvwxyz":
        n = name.lower().count(letter)
        features["count({})".format(letter)] = n
    features["vocals"] = 0

    for letter in "aeiou":
        features["vocals"] = features["vocals"] + 1
    features["consonants"] = 0

    for letter in "bcdfghjklmnpqrstvwxyz":
        features["consonants"] = features["consonants"] + 1

    if chr(features["first_letter"]) in "aeiou":
        features["first_letter_vocal"] = 1
    else:
        features["first_letter_vocal"] = 0

    if chr(features["last_letter"]) in "aeiou":
        features["last_letter_vocal"] = 1
    else:
        features["last_letter_vocal"] = 0

    # h = hyphen.Hyphenator('en_US')
    # features["syllables"] = len(h.syllables(name))
    if ord(name[-1].lower()) == "a":
        features["last_letter_a"] = 1
    else:
        features["last_letter_a"] = 0

    return list(features.values())


def load_dataset(path=ALL_PATH):
    df = pd.read_csv(path)
    df = df[df.first_name.notna()]

    tqdm.pandas()
    X = np.array(df.first_name.progress_apply(lambda x: features_int(x)).tolist())
    y = np.array(df.gender.progress_apply(lambda x: dict(u=0, m=1, f=2)[x]).tolist())

    return X, y


def train_svc():
    X, y = load_dataset()

    model = SVC(probability=True)
    model.fit(X, y)

    logger.info(model.score(X, y))

    pickle.dump(model, open(DATA_DIR / "svc.pkl", "wb"))


def train_forest():
    X, y = load_dataset()

    X, y = make_classification(
        n_samples=7000,
        n_features=33,
        n_informative=33,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )

    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X, y)
    
    logger.info(model.score(X, y))

    with open(DATA_DIR / "forest.pkl", "wb") as fh:
        pickle.dump(model, fh)
