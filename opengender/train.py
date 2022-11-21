import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger

from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics

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
    X = np.array(df.first_name.tolist())
    y = np.array(df.gender.tolist())

    return X, y


def train_svc():
    X, y = load_dataset()

    tv = TfidfVectorizer(analyzer='char')
    clf = LinearSVC(verbose=True,)

    pipeline = Pipeline([('tv', tv), ('clf', clf)])

    params = {
        'tv__strip_accents': (None, 'unicode', 'ascii'),
        # 'tv__token_pattern': (r'(?u)\b\w\w+\b', r'[a-zA-Z]+'),
        'tv__lowercase': (True, False),
        'tv__max_features': np.linspace(10_000, 100_000, 10, dtype='int'),
        # 'tv__max_df': (0.5, 0.75, 1.0),
        # 'tv__min_df': (int(1), 0.01, 0.1),
        'tv__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (2, 3)),
        'clf__C': (0.01, 0.1, 1, 10, 100),
        # 'clf__class_weight': (None, 'balanced'),
        'clf__tol': (1e-4, 1e-5, 1e-6),
    }

    # TODO: f1_macro?
    model = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring='f1_weighted',
        verbose=3,
    )

    # model = LinearSVC()
    model.fit(X, y)

    # model.best_estimator_.named_steps['tv'].stop_words_ = None

    preds = model.predict(X)

    f1 = metrics.f1_score(y, preds, average='weighted')
    accuracy = metrics.accuracy_score(y, preds)
    report = metrics.classification_report(y, preds, output_dict=True)

    logger.info(dict(f1=f1, accuracy=accuracy))
    logger.info(report)

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
