import pickle

from tqdm import tqdm
from typing import Optional

from opengender.dataset import load_damegender
from opengender.paths import CLF_PATH


class Retriever:
    def __init__(self):
        df = load_damegender()
        df = df.set_index("name")
        df = df.drop("count", axis=1)

        tqdm.pandas()
        df = df.rename(columns={'male': 'm', 'female': 'f'})
        df['name'] = df.name.str.lower()
        df["dict"] = df.progress_apply(lambda x: x.to_dict(), axis=1)
        df["gender"] = df.progress_apply(lambda x: max(x.dict, key=x.dict.get), axis=1)
        df["proba"] = df.progress_apply(lambda x: x.dict[x.gender] / 100, axis=1)
        df = df.drop(columns=["m", "f", "dict"])

        self.df = df

    def normalize(self, name: Optional[str]):
        if name:
            # XXX: Add comprehensive normalization
            return name.lower()

    def predict(self, name: Optional[str]):
        """Look up gender and probability of a known name"""
        try:
            return self.df.loc[self.normalize(name)].to_dict()
        except KeyError:
            return dict(gender="u", proba=1.0)


class Classifier:
    def __init__(self, model):
        self.model = model
        self.classes = model.classes_

    @classmethod
    def load(cls, path: str = CLF_PATH):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
            return cls(model)

    def predict(self, name: Optional[str]):
        if name:
            result = self.model.decision_function([name])[0]
            return dict(zip(self.classes, result))
        else:
            return dict(gender="u", proba=1.0)


class OpenGender:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

    @classmethod
    def load(cls):
        retriever = Retriever.load()
        model = Classifier.load()
        return cls(retriever, model)

    def predict(self, name: Optional[str]):
        result = self.retriever.predict(name)

        if result["gender"] == "u":
            result = self.model.predict(name)

        return result
