from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.predictor as predictor_module
from src.predictor import Predictor, ShapExplanationResult


class FakeIndependentMasker:
    def __init__(self, background_data: pd.DataFrame, max_samples: int) -> None:
        self.background_data = background_data
        self.max_samples = max_samples


class FakeShapValues:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> object:
        return self.values[index]


class FakeLinearExplainer:
    def __init__(self, model: object, masker: FakeIndependentMasker) -> None:
        self.model = model
        self.masker = masker

    def __call__(self, features: pd.DataFrame) -> FakeShapValues:
        values = np.array(
            [[10.0, -2.0] for _ in range(len(features))],
            dtype=float,
        )
        return FakeShapValues(values)


@pytest.fixture
def fake_shap(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        maskers=SimpleNamespace(Independent=FakeIndependentMasker),
        LinearExplainer=FakeLinearExplainer,
        plots=SimpleNamespace(
            bar=lambda values: None,
            beeswarm=lambda values: None,
            force=lambda value, matplotlib: None,
            scatter=lambda values: None,
            waterfall=lambda value: None,
        ),
    )
    monkeypatch.setattr(predictor_module, "shap", fake_module)


def test_predict_returns_numpy_array(fitted_linear_model) -> None:
    model, features, _ = fitted_linear_model

    predictions = Predictor(model).predict(features)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(features),)


def test_evaluate_returns_regression_metrics(fitted_linear_model) -> None:
    model, _, _ = fitted_linear_model
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 25.0, 30.0])

    metrics = Predictor(model).evaluate(y_true, y_pred)

    assert metrics == {
        "rmse": pytest.approx(np.sqrt(25 / 3)),
        "r2": pytest.approx(0.875),
    }


def test_explain_returns_shap_values_and_local_feature_contributions(
    fake_shap, fitted_linear_model
) -> None:
    model, features, _ = fitted_linear_model

    explanation = Predictor(model).explain(features, background_data=features)

    assert isinstance(explanation, ShapExplanationResult)
    assert explanation.values.values.shape == features.shape
    assert list(explanation.feature_contributions.columns) == [
        "feature",
        "feature_value",
        "shap_value",
        "absolute_shap_value",
    ]
    assert explanation.feature_contributions["feature"].tolist() == [
        "exam_score",
        "years_exp",
    ]
    assert explanation.feature_contributions["feature_value"].tolist() == [70.0, 2.0]
    assert explanation.feature_contributions["shap_value"].tolist() == [10.0, -2.0]
    assert explanation.feature_contributions["absolute_shap_value"].tolist() == [10.0, 2.0]


def test_explain_can_display_raw_feature_values(fake_shap, fitted_linear_model) -> None:
    model, _, _ = fitted_linear_model
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

    explanation = Predictor(model).explain(
        features,
        background_data=features,
        display_features=display_features,
    )

    assert explanation.feature_contributions["feature_value"].tolist() == [70.0, 2.0]


@pytest.mark.parametrize(
    ("features", "error_type", "message"),
    [
        (np.array([[1.0, 2.0]]), TypeError, "features must be a pandas DataFrame"),
        (pd.DataFrame(), ValueError, "features must contain at least one row"),
    ],
)
def test_explain_rejects_invalid_features(
    fake_shap, fitted_linear_model, features, error_type, message
) -> None:
    model, _, _ = fitted_linear_model

    with pytest.raises(error_type, match=message):
        Predictor(model).explain(features)


def test_explain_rejects_invalid_background_data(fake_shap, fitted_linear_model) -> None:
    model, features, _ = fitted_linear_model

    with pytest.raises(ValueError, match="background_data is missing feature columns"):
        Predictor(model).explain(
            features,
            background_data=pd.DataFrame({"exam_score": [70.0]}),
        )


def test_explain_rejects_mismatched_display_features(fake_shap, fitted_linear_model) -> None:
    model, features, _ = fitted_linear_model

    with pytest.raises(ValueError, match="same row count"):
        Predictor(model).explain(
            features,
            background_data=features,
            display_features=features.iloc[:1],
        )


def test_explain_rejects_unknown_plot_name(fake_shap, fitted_linear_model) -> None:
    model, features, _ = fitted_linear_model

    with pytest.raises(ValueError, match="Unsupported SHAP plot"):
        Predictor(model).explain(features, background_data=features, plot="heatmap")


def test_explain_rejects_out_of_range_sample_index(fake_shap, fitted_linear_model) -> None:
    model, features, _ = fitted_linear_model

    with pytest.raises(IndexError, match="sample_index is outside"):
        Predictor(model).explain(
            features,
            background_data=features,
            sample_index=len(features),
        )
