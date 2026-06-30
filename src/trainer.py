from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Trainer:
    """Train and persist a supported regression model."""

    SUPPORTED_MODELS = {"linear", "lasso"}

    def __init__(self, model_name: str = "linear", alpha: float = 1.0) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            options = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {options}.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        self.model_name = model_name
        self.alpha = alpha
        self.model: LinearRegression | Lasso | None = None

    def _build_model(self) -> LinearRegression | Lasso:
        if self.model_name == "linear":
            return LinearRegression()
        return Lasso(alpha=self.alpha, max_iter=10_000)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression | Lasso:
        """Fit the configured regression model and return it."""
        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        print(f"Trained {self.model_name} regression model.")
        return self.model

    def save_artifact(
        self,
        output_path: str | Path,
        *,
        scaler: StandardScaler | MinMaxScaler | None,
        feature_columns: list[str],
        target_column: str,
    ) -> Path:
        """Save the model and its prediction-time preprocessing contract."""
        if self.model is None:
            raise RuntimeError("Train a model before saving an artifact.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "model_name": self.model_name,
        }
        joblib.dump(artifact, path)
        print(f"Saved model artifact to: {path}")
        return path
