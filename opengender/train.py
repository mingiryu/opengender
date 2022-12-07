import pandas as pd
import pickle

from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics

from opengender import MODEL_PATH
from opengender.build import TRAIN_PATH, TEST_PATH, RANDOM_STATE


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    tv = TfidfVectorizer(
        analyzer="char",
        max_features=50_000,
        ngram_range=(1, 4),
        use_idf=False,
    )
    svc = LinearSVC(
        verbose=True,
        random_state=RANDOM_STATE,
    )

    # Compute probability based on the decision function
    clf = CalibratedClassifierCV(svc)

    model = Pipeline([("tv", tv), ("clf", clf)])

    model.fit(train.X, train.y)

    logger.info(dict(stop_words=len(model.named_steps["tv"].stop_words_)))
    model.named_steps["tv"].stop_words_ = None

    preds = model.predict(test.X)

    f1 = metrics.f1_score(test.y, preds, average="weighted")
    accuracy = metrics.accuracy_score(test.y, preds)

    logger.info(dict(f1=f1, accuracy=accuracy))

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    main()
