import pickle
import numpy as np
import pandas as pd

from loguru import logger

from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from opengender.paths import ALL_PATH, CLF_PATH, INTERALL_PATH


RANDOM_STATE = 7


def encode_train(row):
    if row.male == row.female:
        return "unknown"
    if row.male > row.female:
        return "male"
    else:
        return "female"


def encode_test(x):
    if x.gender == "m":
        return "male"
    if x.gender == "f":
        return "female"
    else:
        return "unknown"


def load_dataset(train_path=INTERALL_PATH, test_path=ALL_PATH, sample_size=400_000):
    train = pd.read_csv(train_path)
    train = train[train.name.notna()]

    train["X"] = train.name.str.lower()
    train["y"] = train.apply(encode_train, axis=1)
    train = train[train.y != "unknown"]
    train = train.sample(sample_size, random_state=RANDOM_STATE)

    test = pd.read_csv(test_path)
    test = test[test.first_name.notna()]

    test["X"] = test.first_name.str.lower()
    test["y"] = test.apply(encode_test, axis=1)
    test = test[test.y != "unknown"]

    return test, train


def main():
    train, test = load_dataset()

    tv = TfidfVectorizer(
        analyzer="char",
        # max_df=0.75,
        # min_df=0.01,
        # max_features=50_000,
        # ngram_range=(1, 3),
        # strip_accents="unicode",
    )
    clf = LinearSVC(
        # kernel="linear",
        # C=10,
        # probability=True,
        verbose=True,
        random_state=RANDOM_STATE,
        # tol=1e-5,
    )

    pipeline = Pipeline([("tv", tv), ("clf", clf)])

    params = {
        "tv__strip_accents": ("unicode", "ascii"),
        "tv__max_features": np.linspace(10_000, 100_000, 10, dtype="int"),
        "tv__max_df": (0.5, 0.75, 0.9),
        "tv__min_df": (0.1, 0.01, 0.001),
        "tv__ngram_range": ((1, 2), (1, 3), (1, 4)),
        "clf__C": (1, 10, 100),
        "clf__tol": (1e-4, 1e-5, 1e-6),
    }

    model = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring="f1_weighted",
        verbose=3,
    )

    model.fit(train.X, train.y)

    model.best_estimator_.named_steps["tv"].stop_words_ = None

    logger.info(dict(best=model.best_estimator_))

    preds = model.predict(test.X)

    f1 = metrics.f1_score(test.y, preds, average="weighted")
    accuracy = metrics.accuracy_score(test.y, preds)
    report = metrics.classification_report(test.y, preds, output_dict=True)

    logger.info(dict(f1=f1, accuracy=accuracy))
    logger.info(report)

    with open(CLF_PATH, "wb") as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    main()
