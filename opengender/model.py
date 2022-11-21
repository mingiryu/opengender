import pandas as pd
import pickle

from typing import Optional
from tqdm import tqdm

from opengender.train import features_int
from opengender.paths import DATA_DIR, INTERALL_PATH


class Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    @classmethod
    def load(cls, path: str = INTERALL_PATH):
        df = pd.read_csv(path)
        df = df.set_index("name")
        df = df.drop("count", axis=1)

        tqdm.pandas()
        df["dict"] = df.progress_apply(lambda x: x.to_dict(), axis=1)
        df["gender"] = df.progress_apply(lambda x: max(x.dict, key=x.dict.get), axis=1)
        df["proba"] = df.progress_apply(lambda x: x.dict[x.gender] / 100, axis=1)
        df = df.drop(columns=["male", "female", "dict"])

        return cls(df)

    def normalize(self, name: Optional[str]):
        if name:
            # XXX: Add comprehensive normalization
            return name.upper()

    def predict(self, name: Optional[str]):
        """Look up gender and probability of a known name"""
        try:
            return self.df.loc[self.normalize(name)].to_dict()
        except KeyError:
            return dict(gender="unknown", proba=1.0)


class Classifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
            return cls(model)

    def predict(self, name: Optional[str]):
        if name:
            return self.model.predict_proba([features_int(name)])
        else:
            return dict(gender="unknown", proba=1.0)
