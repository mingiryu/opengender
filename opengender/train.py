import pickle

from loguru import logger

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from opengender import metrics
from opengender.paths import CLF_PATH
from opengender.dataset import (
    load_name_gender_inference,
    load_damegender,
    load_wiki_gendersort,
)

RANDOM_STATE = 7


def main():
    train = load_name_gender_inference()
    test = load_name_gender_inference()

    tv = TfidfVectorizer(
        analyzer="char",
        # max_df=0.75,
        # min_df=0.01,
        # max_features=10_000,
        ngram_range=(1, 4),
    )
    clf = RandomForestClassifier(
        # kernel="linear",
        # C=10,
        # probability=True,
        verbose=True,
        random_state=RANDOM_STATE,
        # tol=1e-5,
    )

    pipeline = Pipeline([("tv", tv), ("clf", clf)])

    params = {
        # "tv__max_features": np.linspace(10_000, 100_000, 10, dtype="int"),
        # "tv__max_df": (0.5, 0.75, 0.9),
        # "tv__min_df": (0.1, 0.01, 0.001),
        # "tv__ngram_range": ((1, 2), (1, 3), (1, 4)),
        # "clf__C": (1, 10, 100),
        # "clf__tol": (1e-4, 1e-5, 1e-6),
    }

    model = HalvingRandomSearchCV(
        pipeline,
        params,
        scoring=make_scorer(metrics.error_coded, greater_is_better=False),
        verbose=3,
    )

    model.fit(train.X, train.y)

    model.best_estimator_.named_steps["tv"].stop_words_ = None

    logger.info(dict(best=model.best_estimator_))

    preds = model.predict(test.X)

    logger.info(
        dict(
            error_coded=metrics.error_coded(test.y, preds),
            error_coded_without_na=metrics.error_coded_without_na(test.y, preds),
            na_coded=metrics.na_coded(test.y, preds),
            error_gender_bias=metrics.error_gender_bias(test.y, preds),
        )
    )

    with open(CLF_PATH, "wb") as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    main()
