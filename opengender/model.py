import pandas as pd

from typing import Optional
from tqdm import tqdm

from opengender.paths import INTERALL_PATH


class Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    @classmethod
    def load_model(cls, path: str = INTERALL_PATH):
        df = pd.read_csv(path)
        df = df.set_index("name")
        df = df.drop("count", axis=1)

        tqdm.pandas()
        df["dict"] = df.progress_apply(lambda x: x.to_dict(), axis=1)
        df["gender"] = df.progress_apply(lambda x: max(x.dict, key=x.dict.get), axis=1)
        df["proba"] = df.progress_apply(lambda x: x.dict[x.gender] / 100, axis=1)
        df = df.drop(columns=["male", "female", "dict"])

        return cls(df)

    def normalize(self, text: Optional[str]):
        if text:
            # XXX: Add comprehensive text normalization
            return text.upper()

    def predict(self, name: Optional[str]):
        """Look up gender and probability of a known name"""
        try:
            return self.df.loc[self.normalize(name)].to_dict()
        except KeyError:
            return dict(gender="unknown", proba=1.0)
