import joblib
import pandas as pd
import pytest

from src.trainer import Trainer


def test_trainer_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        Trainer(model_name="ridge")


def test_trainer_saves_model_and_prediction_contract(tmp_path):
    X_train = pd.DataFrame(
        {"years_exp": [1, 2, 3, 4], "exam_score": [50, 60, 70, 80]}
    )
    y_train = pd.Series([30_000, 40_000, 50_000, 60_000], name="salary")
    trainer = Trainer(model_name="linear")
    trainer.train(X_train, y_train)

    output_path = trainer.save_artifact(
        tmp_path / "artifacts" / "linear.pkl",
        scaler=None,
        feature_columns=["years_exp", "exam_score"],
        target_column="salary",
    )
    artifact = joblib.load(output_path)

    assert output_path.is_file()
    assert artifact["model_name"] == "linear"
    assert artifact["feature_columns"] == ["years_exp", "exam_score"]
    assert artifact["target_column"] == "salary"
    assert artifact["scaler"] is None
