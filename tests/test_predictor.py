import pandas as pd
from sklearn.linear_model import LinearRegression

from src.predictor import Predictor, ShapExplanationResult


def test_explain_returns_shap_values_and_feature_importance() -> None:
    features = pd.DataFrame(
        {
            "exam_score": [70.0, 80.0, 90.0, 60.0],
            "years_exp": [2.0, 4.0, 6.0, 1.0],
        }
    )
    target = pd.Series([40_000.0, 55_000.0, 70_000.0, 34_000.0])
    model = LinearRegression().fit(features, target)

    explanation = Predictor(model).explain(features, background_data=features)

    assert isinstance(explanation, ShapExplanationResult)
    assert explanation.values.values.shape == features.shape
    assert list(explanation.feature_importance.columns) == [
        "feature",
        "mean_absolute_shap_value",
    ]
    assert set(explanation.feature_importance["feature"]) == set(features.columns)
