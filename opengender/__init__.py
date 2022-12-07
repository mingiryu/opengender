import numpy as np
import pickle

from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"


class OpenGender:
    def __init__(self):
        with open(MODEL_PATH, "rb") as fh:
            self.model = pickle.load(fh)

    def predict(self, name: Optional[str]):
        proba = self.model.predict_proba([name])[0]
        idx = np.argmax(proba)
        return dict(gender=self.model.classes_[idx], proba=proba[idx])

    def __call__(self, name: Optional[str]):
        if not name:
            return dict(gender="u", proba=1.0)

        return self.predict(name)
