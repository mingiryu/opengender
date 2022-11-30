import pytest

from opengender.metrics import (
    components,
    error_coded,
    error_coded_without_na,
    errror_gender_bias,
    na_coded,
)


@pytest.mark.parametrize(
    "y_true,y_pred,expect",
    [
        # Sanity check
        (("male", "male"), ("male", "female"), (1, 1, 0, 0, 0, 0)),
        # Unknown in prediction
        (["male", "male", "female"], ["male", "female", "unknown"], (1, 1, 0, 0, 0, 1)),
        # Unknown in label
        (["male", "male", "unknown"], ["male", "female", "female"], (1, 1, 0, 0, 0, 0)),
    ],
)
def test_components(y_true, y_pred, expect):
    assert components(y_true, y_pred) == expect
