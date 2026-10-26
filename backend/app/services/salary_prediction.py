from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import joblib  # Preserve the existing artifact-loader patch point.
import pandas as pd

# Preserve `cd backend; uvicorn app.main:app` in this source-tree project,
# where the repository's src package is not installed as a distribution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import inference, predictor

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "linear_regression.pkl"
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "salary_data.xlsx"
COLUMN_RENAMES = {
    "#": "index",
    "Exam Score (0-100)": "exam_score",
    "Exam Score (0–100)": "exam_score",
    "Exam Score (0â€“100)": "exam_score",
    "Exam Score (0Ã¢â‚¬â€œ100)": "exam_score",
    "Years of Experience": "years_exp",
    "Salary (€)": "salary",
    "Salary (â‚¬)": "salary",
    "Salary (Ã¢â€šÂ¬)": "salary",
}


@lru_cache
def load_model_artifact() -> dict[str, Any]:
    return inference.load_model_artifact(MODEL_PATH)


def predict_salary(*, exam_score: int, years_exp: int) -> float:
    artifact = load_model_artifact()
    model_features = build_model_features(
        artifact,
        exam_score=exam_score,
        years_exp=years_exp,
    )

    return float(predictor.Predictor(artifact["model"]).predict(model_features)[0])


def build_raw_features(
    feature_columns: list[str], *, exam_score: int, years_exp: int
) -> pd.DataFrame:
    return inference.build_raw_features(
        feature_columns, exam_score=exam_score, years_exp=years_exp
    )


def build_model_features(
    artifact: dict[str, Any], *, exam_score: int, years_exp: int
) -> pd.DataFrame:
    _, model_features = inference.prepare_prediction_features(
        feature_columns=artifact["feature_columns"],
        exam_score=exam_score,
        years_exp=years_exp,
        scaler=artifact.get("scaler"),
    )
    return model_features


@lru_cache
def load_background_features() -> pd.DataFrame:
    artifact = load_model_artifact()
    feature_columns = artifact["feature_columns"]
    data = pd.read_excel(DATA_PATH).rename(columns=COLUMN_RENAMES)
    background_features = data.dropna().loc[:, feature_columns]
    return inference.transform_features(
        background_features, feature_columns=feature_columns, scaler=artifact.get("scaler")
    )


def explain_salary_prediction(
    *, exam_score: int, years_exp: int, predicted_salary: float
) -> dict[str, Any]:
    artifact = load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    raw_features = build_raw_features(
        feature_columns,
        exam_score=exam_score,
        years_exp=years_exp,
    )
    model_features = build_model_features(
        artifact,
        exam_score=exam_score,
        years_exp=years_exp,
    )
    background_features = load_background_features()

    shap_values = predictor.calculate_shap_values(
        model,
        model_features,
        background_data=background_features,
        max_background_samples=118,
    )
    values = shap_values.values[0]
    base_value = shap_values.base_values[0]

    contributions = []
    for feature, feature_value, shap_value in zip(
        feature_columns,
        raw_features.iloc[0].tolist(),
        values,
    ):
        contributions.append(
            {
                "feature": feature,
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
                "absolute_shap_value": abs(float(shap_value)),
            }
        )

    contributions.sort(key=lambda item: item["absolute_shap_value"], reverse=True)

    return {
        "predicted_salary": predicted_salary,
        "base_value": float(base_value),
        "contributions": contributions,
    }
