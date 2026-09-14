from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap

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
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        return {
            "model": artifact,
            "scaler": None,
            "feature_columns": ["exam_score", "years_exp"],
        }

    required_keys = {"model", "feature_columns"}
    missing_keys = required_keys - artifact.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Model artifact is missing required keys: {missing}")

    return artifact


def predict_salary(*, exam_score: int, years_exp: int) -> float:
    artifact = load_model_artifact()
    model_features = build_model_features(
        artifact,
        exam_score=exam_score,
        years_exp=years_exp,
    )

    return float(artifact["model"].predict(model_features)[0])


def build_raw_features(
    feature_columns: list[str], *, exam_score: int, years_exp: int
) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exam_score": exam_score, "years_exp": years_exp}],
        columns=feature_columns,
    )


def build_model_features(
    artifact: dict[str, Any], *, exam_score: int, years_exp: int
) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    raw_features = build_raw_features(
        feature_columns,
        exam_score=exam_score,
        years_exp=years_exp,
    )
    scaler = artifact.get("scaler")

    if scaler is not None:
        return pd.DataFrame(
            scaler.transform(raw_features),
            columns=feature_columns,
        )

    return raw_features


@lru_cache
def load_background_features() -> pd.DataFrame:
    artifact = load_model_artifact()
    feature_columns = artifact["feature_columns"]
    data = pd.read_excel(DATA_PATH).rename(columns=COLUMN_RENAMES)
    background_features = data.dropna().loc[:, feature_columns]
    scaler = artifact.get("scaler")

    if scaler is not None:
        return pd.DataFrame(
            scaler.transform(background_features),
            columns=feature_columns,
        )

    return background_features


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

    masker = shap.maskers.Independent(background_features, max_samples=118)
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(model_features)
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
