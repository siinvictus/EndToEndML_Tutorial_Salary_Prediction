"""Artifact loading and feature preparation shared by inference callers.

Callers own artifact paths, caching, and background-data selection. Training
continues to fit its own model and scaler through the existing pipeline.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model_artifact(model_path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
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


def build_raw_features(
    feature_columns: list[str], *, exam_score: float, years_exp: float
) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exam_score": exam_score, "years_exp": years_exp}],
        columns=feature_columns,
    )


def transform_features(
    raw_features: pd.DataFrame, *, feature_columns: list[str], scaler: Any = None
) -> pd.DataFrame:
    if scaler is not None:
        return pd.DataFrame(
            scaler.transform(raw_features),
            columns=feature_columns,
        )
    return raw_features


def prepare_prediction_features(
    *, feature_columns: list[str], exam_score: float, years_exp: float,
    scaler: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_features = build_raw_features(
        feature_columns, exam_score=exam_score, years_exp=years_exp
    )
    model_features = transform_features(
        raw_features, feature_columns=feature_columns, scaler=scaler
    )
    return raw_features, model_features
