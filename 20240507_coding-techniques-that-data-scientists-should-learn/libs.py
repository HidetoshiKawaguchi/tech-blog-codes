import pandas as pd


def remove_missing_and_outlier(
    df: pd.DataFrame,
    how: str = "any",
    year_birth_lower: int = 1920,
    income_upper: int = 666666,
) -> pd.DataFrame:
    df = df.dropna(how=how)
    df = df[(df["Year_Birth"] > year_birth_lower) & (df["Income"] < income_upper)]
    return df
