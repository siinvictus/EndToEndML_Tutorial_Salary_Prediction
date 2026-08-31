import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def salary_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exam_score": [70.0, 80.0, 90.0, 60.0, 75.0],
            "years_exp": [2.0, 4.0, 6.0, 1.0, 3.0],
            "salary": [40_000.0, 55_000.0, 70_000.0, 34_000.0, 48_000.0],
        }
    )


@pytest.fixture
def fitted_linear_model(salary_dataframe: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, pd.Series]:
    features = salary_dataframe[["exam_score", "years_exp"]]
    target = salary_dataframe["salary"]
    model = LinearRegression().fit(features, target)
    return model, features, target
