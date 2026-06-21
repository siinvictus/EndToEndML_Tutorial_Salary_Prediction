import numpy as np
import pandas as pd

from src.preprocessor import Preprocessor


def test_preprocess_renames_columns_excludes_index_and_scales_training_data():
    raw_data = pd.DataFrame(
        {
            "#": range(1, 11),
            "Exam Score (0–100)": np.linspace(50, 95, 10),
            "Years of Experience": np.arange(1, 11, dtype=float),
            "Salary (€)": np.linspace(40_000, 100_000, 10),
        }
    )
    raw_data.loc[0, "Years of Experience"] = np.nan
    preprocessor = Preprocessor(scaling="standard", test_size=0.25, random_state=7)

    X_train, X_test, y_train, y_test = preprocessor.preprocess(raw_data)

    assert preprocessor.feature_columns == ["exam_score", "years_exp"]
    assert list(X_train.columns) == ["exam_score", "years_exp"]
    assert list(X_test.columns) == ["exam_score", "years_exp"]
    assert len(X_train) == 6
    assert len(X_test) == 3
    assert len(y_train) == 6
    assert len(y_test) == 3
    assert preprocessor.scaler is not None
    assert np.allclose(X_train.mean(), 0.0)


def test_preprocess_can_skip_scaling():
    data = pd.DataFrame(
        {
            "index": [1, 2, 3, 4],
            "years_exp": [1, 2, 3, 4],
            "salary": [20, 30, 40, 50],
        }
    )
    preprocessor = Preprocessor(scaling="none", test_size=0.5, random_state=42)

    X_train, _, _, _ = preprocessor.preprocess(data)

    assert preprocessor.scaler is None
    assert list(X_train.columns) == ["years_exp"]
