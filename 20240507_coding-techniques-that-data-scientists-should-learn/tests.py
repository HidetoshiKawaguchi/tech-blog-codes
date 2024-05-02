import pandas as pd
from libs import remove_missing_and_outlier


def test_remove_missing_and_outlier():
    df = pd.DataFrame(
        {
            "Year_Birth": [1921, 1920, 2000, 2000, 2000, 2000],
            "Income": [100000, 100000, 666665, 666666, 100000, 100000],
            "Any": [True, True, True, True, True, None],
        }
    )
    df = remove_missing_and_outlier(
        df, how="any", year_birth_lower=1920, income_upper=666666
    )
    assert len(df) == 3
    assert df.index[0] == 0
    assert df.index[1] == 2
    assert df.index[2] == 4
