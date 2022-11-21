import pytest

from opengender.model import Retriever


@pytest.fixture(scope="package")
def retriever():
    yield Retriever.load()


@pytest.mark.parametrize(
    "test,expect",
    [
        # Sanity checks
        ("David", {"gender": "male", "proba": 0.9973847215987144}),
        ("Mary", {"gender": "female", "proba": 0.996117197103305}),
        # Unknown name
        ("Mingi", {"gender": "unknown", "proba": 1.0}),
        # Invalid inputs
        ("", {"gender": "unknown", "proba": 1.0}),
        (None, {"gender": "unknown", "proba": 1.0}),
    ],
)
def test_retriver_predict(retriever, test, expect):
    assert retriever.predict(test) == expect
