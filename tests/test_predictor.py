import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.predictor import Predictor, ShapExplanationResult


def test_explain_returns_shap_values_and_local_feature_contributions() -> None:
    features = pd.DataFrame(
        {
            "exam_score": [70.0, 80.0, 90.0, 60.0],
            "years_exp": [2.0, 4.0, 6.0, 1.0],
        }
    )
    target = pd.Series([40_000.0, 55_000.0, 70_000.0, 34_000.0])
    model = LinearRegression().fit(features, target)
    return model, features, target


def test_explain_returns_shap_values_and_feature_importance(fitted_model) -> None:
    model, features, target = fitted_model
    
    explanation = Predictor(model).explain(features, background_data=features)

    assert isinstance(explanation, ShapExplanationResult)
    assert explanation.values.values.shape == features.shape
    assert list(explanation.feature_contributions.columns) == [
        "feature",
        "feature_value",
        "shap_value",
        "absolute_shap_value",
    ]
    assert set(explanation.feature_contributions["feature"]) == set(features.columns)
    assert set(explanation.feature_contributions["feature_value"]) == set(
        features.iloc[0]
    )
    assert set(explanation.feature_contributions["shap_value"]) == set(
        explanation.values.values[0]
    )


def test_explain_can_display_raw_feature_values() -> None:
    features = pd.DataFrame(
        {
            "exam_score": [0.0, 1.0, 2.0],
            "years_exp": [0.0, 1.0, 2.0],
        }
    )
    display_features = pd.DataFrame(
        {
            "exam_score": [70.0, 80.0, 90.0],
            "years_exp": [2.0, 4.0, 6.0],
        }
    )
    target = pd.Series([40_000.0, 55_000.0, 70_000.0])
    model = LinearRegression().fit(features, target)

    explanation = Predictor(model).explain(
        features,
        background_data=features,
        display_features=display_features,
    )

    assert set(explanation.feature_contributions["feature_value"]) == set(
        display_features.iloc[0]
    )
