from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import shap

@dataclass(frozen=True)
class ShapExplanationResult:
    """Container for model explanations computed with SHAP."""

    values: object
    feature_contributions: pd.DataFrame


class Predictor:
    """Generate predictions and evaluate a trained regression model."""

    SUPPORTED_EXPLANATION_PLOTS = {"bar", "beeswarm", "force", "scatter", "waterfall"}

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

    def explain(
        self,
        features: pd.DataFrame,
        *,
        background_data: pd.DataFrame | None = None,
        max_background_samples: int = 118,
        plot: str | None = None,
        sample_index: int = 0,
        display_features: pd.DataFrame | None = None,
    ) -> ShapExplanationResult:
        """Explain model predictions with SHAP values.

        The notebook computes SHAP values with ``shap.LinearExplainer`` and an
        independent masker. This method keeps that same approach, but returns
        reusable data instead of making plotting a required side effect. The
        returned contribution table explains one selected prediction row, not
        an average across all rows.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame with named columns.")
        if features.empty:
            raise ValueError("features must contain at least one row.")
        if max_background_samples <= 0:
            raise ValueError("max_background_samples must be a positive integer.")
        if sample_index < 0 or sample_index >= len(features):
            raise IndexError("sample_index is outside the SHAP explanation range.")
        if display_features is not None:
            if not isinstance(display_features, pd.DataFrame):
                raise TypeError("display_features must be a pandas DataFrame when provided.")
            if len(display_features) != len(features):
                raise ValueError("display_features must have the same row count as features.")
            missing_display_columns = set(features.columns) - set(display_features.columns)
            if missing_display_columns:
                columns = ", ".join(sorted(missing_display_columns))
                raise ValueError(f"display_features is missing feature columns: {columns}.")
            display_features = display_features.loc[:, features.columns]

        if background_data is None:
            background_data = features
        if not isinstance(background_data, pd.DataFrame):
            raise TypeError("background_data must be a pandas DataFrame when provided.")
        if background_data.empty:
            raise ValueError("background_data must contain at least one row.")

        missing_columns = set(features.columns) - set(background_data.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"background_data is missing feature columns: {columns}.")

        background_data = background_data.loc[:, features.columns]

     

        masker = shap.maskers.Independent(
            background_data, max_samples=max_background_samples
        )
       
        explainer = shap.LinearExplainer(self.model, masker)
        shap_values = explainer(features)

        selected_values = np.asarray(shap_values.values[sample_index])
        selected_features = (
            display_features.iloc[sample_index]
            if display_features is not None
            else features.iloc[sample_index]
        )
        contributions = pd.DataFrame(
            {
                "feature": list(features.columns),
                "feature_value": selected_features.to_numpy(),
                "shap_value": selected_values,
                "absolute_shap_value": np.abs(selected_values),
            }
        ).sort_values("absolute_shap_value", ascending=False, ignore_index=True)

        if plot is not None:
            self._plot_shap_explanation(shap_values, plot=plot, sample_index=sample_index)

        return ShapExplanationResult(
            values=shap_values, feature_contributions=contributions
        )

    def _plot_shap_explanation(
        self, shap_values: object, *, plot: str, sample_index: int
    ) -> None:
        """Render one supported SHAP plot."""
        if plot not in self.SUPPORTED_EXPLANATION_PLOTS:
            options = ", ".join(sorted(self.SUPPORTED_EXPLANATION_PLOTS))
            raise ValueError(f"Unsupported SHAP plot '{plot}'. Choose one of: {options}.")
        if sample_index < 0 or sample_index >= len(shap_values):
            raise IndexError("sample_index is outside the SHAP explanation range.")

        import shap

        if plot == "bar":
            shap.plots.bar(shap_values)
        elif plot == "beeswarm":
            shap.plots.beeswarm(shap_values)
        elif plot == "force":
            shap.plots.force(shap_values[sample_index], matplotlib=True)
        elif plot == "scatter":
            shap.plots.scatter(shap_values)
        else:
            shap.plots.waterfall(shap_values[sample_index])
