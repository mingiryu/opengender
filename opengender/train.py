import pickle
import numpy as np
import pandas as pd

from loguru import logger

from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from opengender.paths import ALL_PATH, DATA_DIR, INTERALL_PATH


def encode(row):
    if row.male == row.female:
        return "u"
    elif row.male > row.female:
        return "m"
    else:
        return "f"


def load_dataset(train_path=INTERALL_PATH, test_path=ALL_PATH):
    train = pd.read_csv(train_path)
    train = train[train.name.notna()]

    train["X"] = train.name.str.lower()
    train["y"] = train.apply(encode, axis=1)

    test = pd.read_csv(test_path)
    test = test[test.first_name.notna()]

    test["X"] = test.first_name.str.lower()
    test["y"] = test.gender.copy()

    return train, test


def main():
    train, test = load_dataset()

    tv = TfidfVectorizer(analyzer="char")
    clf = LinearSVC(verbose=True)

    pipeline = Pipeline([("tv", tv), ("clf", clf)])

    params = {
        "tv__strip_accents": (None, "unicode", "ascii"),
        "tv__max_features": np.linspace(10_000, 100_000, 10, dtype="int"),
        "tv__max_df": (0.5, 0.75, 1.0),
        "tv__min_df": (int(1), 0.01, 0.1),
        "tv__ngram_range": ((1, 1), (1, 2), (1, 3), (1, 4), (2, 3)),
        "clf__C": (0.01, 0.1, 1, 10, 100),
        "clf__class_weight": (None, "balanced"),
        "clf__tol": (1e-4, 1e-5, 1e-6),
    }

    model = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring="f1_weighted",
        verbose=3,
    )

    # model = LinearSVC()
    model.fit(train.X, train.y)

    model.best_estimator_.named_steps['tv'].stop_words_ = None

    preds = model.predict(test.X)

    f1 = metrics.f1_score(test.y, preds, average="weighted")
    accuracy = metrics.accuracy_score(test.y, preds)
    report = metrics.classification_report(test.y, preds, output_dict=True)

    logger.info(dict(f1=f1, accuracy=accuracy))
    logger.info(report)

    pickle.dump(model, open(DATA_DIR / "model.pkl", "wb"))


if __name__ == "__main__":
    main()
