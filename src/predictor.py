import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


class Predictor:
    """Generate predictions and evaluate a trained regression model."""

    def __init__(self, model: LinearRegression | Lasso) -> None:
        self.model = model

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target values for preprocessed features."""
        return self.model.predict(X_test)

    def evaluate(
        self, y_test: pd.Series | np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate and print regression quality metrics."""
        metrics = {
            "rmse": float(root_mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.4f}")
        return metrics
