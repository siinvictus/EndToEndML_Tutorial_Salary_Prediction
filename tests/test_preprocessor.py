import pandas as pd
import pytest

from src.preprocessor import Preprocessor


def test_preprocess_renames_columns_drops_nulls_splits_and_scales() -> None:
    raw_data = pd.DataFrame(
        {
            "#": [1, 2, 3, 4, 5],
            "Exam Score (0â€“100)": [70.0, 80.0, 90.0, None, 75.0],
            "Years of Experience": [2.0, 4.0, 6.0, 8.0, 3.0],
            "Salary (â‚¬)": [40_000.0, 55_000.0, 70_000.0, 80_000.0, 48_000.0],
        }
    )
    preprocessor = Preprocessor(test_size=0.25, random_state=0)

    X_train, X_test, y_train, y_test = preprocessor.preprocess(raw_data)

    assert preprocessor.feature_columns == ["exam_score", "years_exp"]
    assert preprocessor.scaler is not None
    assert list(X_train.columns) == ["exam_score", "years_exp"]
    assert list(X_test.columns) == ["exam_score", "years_exp"]
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
    assert X_train.mean().abs().max() == pytest.approx(0.0)


@pytest.mark.parametrize("scaling", ["standard", "minmax", "none"])
def test_preprocess_supports_all_scaling_options(
    scaling: str, salary_dataframe: pd.DataFrame
) -> None:
    preprocessor = Preprocessor(scaling=scaling, test_size=0.4, random_state=42)

    X_train, X_test, _, _ = preprocessor.preprocess(salary_dataframe)

    assert list(X_train.columns) == ["exam_score", "years_exp"]
    assert list(X_test.columns) == ["exam_score", "years_exp"]
    if scaling == "none":
        assert preprocessor.scaler is None
    else:
        assert preprocessor.scaler is not None


def test_minmax_scaling_fits_on_training_data_only(salary_dataframe: pd.DataFrame) -> None:
    preprocessor = Preprocessor(scaling="minmax", test_size=0.4, random_state=42)

    X_train, X_test, _, _ = preprocessor.preprocess(salary_dataframe)

    assert X_train.min().min() == pytest.approx(0.0)
    assert X_train.max().max() == pytest.approx(1.0)
    assert list(X_test.columns) == preprocessor.feature_columns


def test_split_features_target_rejects_missing_target(salary_dataframe: pd.DataFrame) -> None:
    preprocessor = Preprocessor(target_column="missing_target")

    with pytest.raises(ValueError, match="Target column 'missing_target' was not found"):
        preprocessor.split_features_target(salary_dataframe)


def test_split_features_target_rejects_non_numeric_target() -> None:
    data = pd.DataFrame(
        {
            "exam_score": [70, 80],
            "years_exp": [2, 4],
            "salary": ["low", "high"],
        }
    )

    with pytest.raises(ValueError, match="must be numeric"):
        Preprocessor().split_features_target(data)


def test_split_features_target_rejects_data_without_numeric_features() -> None:
    data = pd.DataFrame(
        {
            "department": ["Engineering", "Finance"],
            "salary": [40_000.0, 55_000.0],
        }
    )

    with pytest.raises(ValueError, match="No numeric feature columns"):
        Preprocessor().split_features_target(data)


@pytest.mark.parametrize("bad_scaling", ["robust", "", "STANDARD"])
def test_constructor_rejects_unsupported_scaling(bad_scaling: str) -> None:
    with pytest.raises(ValueError, match="Unsupported scaling"):
        Preprocessor(scaling=bad_scaling)


@pytest.mark.parametrize("bad_test_size", [0, 1, -0.1, 1.1])
def test_constructor_rejects_invalid_test_size(bad_test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size must be greater than 0"):
        Preprocessor(test_size=bad_test_size)
