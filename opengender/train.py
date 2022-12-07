import pandas as pd
import pickle

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from opengender import metrics
from opengender.paths import CLF_PATH
from opengender.dataset import (
    load_name_gender_inference,
    load_damegender,
    load_wiki_gendersort,
)


RANDOM_STATE = 7


def main():
    dame = load_damegender()[['X', 'y']]
    wiki = load_wiki_gendersort()[['X', 'y']]

    train = pd.concat([dame, wiki])
    test = load_name_gender_inference()[['X', 'y']]

    # Make sure test data is unseen during training
    train = train[~train.X.isin(test.X)]

    tv = TfidfVectorizer(
        analyzer="char",
        max_features=50_000,
        ngram_range=(1, 4),
        use_idf=False,
    )
    clf = LinearSVC(
        verbose=True,
        random_state=RANDOM_STATE,
    )

    model = Pipeline([("tv", tv), ("clf", clf)])

    model.fit(train.X, train.y)

    logger.info(dict(stop_words=len(model.named_steps["tv"].stop_words_)))
    model.named_steps["tv"].stop_words_ = None

    pred = model.predict(test.X)

    logger.info(
        dict(
            error_coded=metrics.error_coded(test.y, pred),
            error_coded_without_na=metrics.error_coded_without_na(test.y, pred),
            na_coded=metrics.na_coded(test.y, pred),
            error_gender_bias=metrics.error_gender_bias(test.y, pred),
        )
    )

    with open(CLF_PATH, "wb") as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    main()
