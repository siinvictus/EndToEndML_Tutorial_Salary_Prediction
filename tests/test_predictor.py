import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.predictor import Predictor, ShapExplanationResult


@pytest.fixture
def fitted_model():
    features = pd.DataFrame({
        "exam_score": [70.0, 80.0, 90.0, 60.0],
        "years_exp": [2.0, 4.0, 6.0, 1.0],
    })
    target = pd.Series([40_000.0, 55_000.0, 70_000.0, 34_000.0])
    model = LinearRegression().fit(features, target)
    return model, features, target


def test_explain_returns_shap_values_and_feature_importance(fitted_model) -> None:
    model, features, target = fitted_model
    
    explanation = Predictor(model).explain(features, background_data=features)

    assert isinstance(explanation, ShapExplanationResult)
    assert explanation.values.values.shape == features.shape
    assert list(explanation.feature_importance.columns) == [
        "feature",
        "mean_absolute_shap_value",
    ]
    assert set(explanation.feature_importance["feature"]) == set(features.columns)


def test_predict_returns_array(fitted_model) -> None:
    model, features, target = fitted_model
    predictor = Predictor(model)
    predictions = predictor.predict(features)
    assert len(predictions) == len(features)


def test_evaluate_returns_rmse_and_r2(fitted_model) -> None:
    model, features, target = fitted_model
    predictor = Predictor(model)
    predictions = predictor.predict(features)
    metrics = predictor.evaluate(target, predictions)
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0