import pytest

from opengender import OpenGender


@pytest.fixture(scope="package")
def gender():
    yield OpenGender()


@pytest.mark.parametrize(
    "test,expect",
    [
        # Sanity checks
        ("David", {"gender": "m", "proba": 0.9446712224321301}),
        ("Mary", {"gender": "f", "proba": 0.9830306722011564}),
        # Unknown name
        ("Mingi", {"gender": "m", "proba": 0.837469101587949}),
        # Invalid inputs
        ("", {"gender": "u", "proba": 1.0}),
        (None, {"gender": "u", "proba": 1.0}),
    ],
)
def test_model(gender, test, expect):
    assert gender(test) == expect
