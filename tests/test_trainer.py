import joblib
import pytest
from sklearn.linear_model import Lasso, LinearRegression

from src.trainer import Trainer


def test_train_builds_linear_regression_model(salary_dataframe) -> None:
    trainer = Trainer(model_name="linear")
    features = salary_dataframe[["exam_score", "years_exp"]]
    target = salary_dataframe["salary"]

    model = trainer.train(features, target)

    assert isinstance(model, LinearRegression)
    assert trainer.model is model
    assert model.coef_.shape == (2,)


def test_train_builds_lasso_model_with_configured_alpha(salary_dataframe) -> None:
    trainer = Trainer(model_name="lasso", alpha=0.25)
    features = salary_dataframe[["exam_score", "years_exp"]]
    target = salary_dataframe["salary"]

    model = trainer.train(features, target)

    assert isinstance(model, Lasso)
    assert model.alpha == 0.25
    assert trainer.model is model


def test_save_artifact_persists_model_contract(tmp_path, salary_dataframe) -> None:
    trainer = Trainer(model_name="linear")
    features = salary_dataframe[["exam_score", "years_exp"]]
    target = salary_dataframe["salary"]
    trainer.train(features, target)
    output_path = tmp_path / "nested" / "linear_regression.pkl"

    saved_path = trainer.save_artifact(
        output_path,
        scaler=None,
        feature_columns=["exam_score", "years_exp"],
        target_column="salary",
    )

    assert saved_path == output_path
    assert output_path.exists()
    artifact = joblib.load(output_path)
    assert isinstance(artifact["model"], LinearRegression)
    assert artifact["scaler"] is None
    assert artifact["feature_columns"] == ["exam_score", "years_exp"]
    assert artifact["target_column"] == "salary"
    assert artifact["model_name"] == "linear"


def test_save_artifact_rejects_untrained_model(tmp_path) -> None:
    trainer = Trainer(model_name="linear")

    with pytest.raises(RuntimeError, match="Train a model before saving"):
        trainer.save_artifact(
            tmp_path / "model.pkl",
            scaler=None,
            feature_columns=["exam_score"],
            target_column="salary",
        )


@pytest.mark.parametrize("bad_model", ["ridge", "", "Linear"])
def test_constructor_rejects_unsupported_model_name(bad_model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        Trainer(model_name=bad_model)


def test_constructor_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        Trainer(model_name="lasso", alpha=-0.1)
