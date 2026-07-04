import argparse
from pathlib import Path

import pandas as pd

from src.dataloader import DataLoader
from src.predictor import Predictor
from src.preprocessor import Preprocessor
from src.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the regression pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a salary regression model."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV or Excel dataset.")
    parser.add_argument(
        "--model", choices=("linear", "lasso"), default="linear", help="Model to train."
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Lasso alpha value.")
    parser.add_argument(
        "--scaling",
        choices=("standard", "minmax", "none"),
        default="standard",
        help="Feature scaling strategy.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-set proportion.")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for the train/test split."
    )
    parser.add_argument(
        "--model-output", help="Path for the saved model artifact (.pkl)."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print SHAP feature contributions for one prediction.",
    )
    parser.add_argument(
        "--exam-score",
        type=float,
        help="Exam score for a new salary prediction.",
    )
    parser.add_argument(
        "--years-exp",
        type=float,
        help="Years of experience for a new salary prediction.",
    )
    return parser


def default_model_output(model_name: str) -> Path:
    """Return the standard artifact path for a model type."""
    filename = "linear_regression.pkl" if model_name == "linear" else "lasso_regression.pkl"
    return Path("models") / filename


def has_prediction_input(args: argparse.Namespace) -> bool:
    """Return whether the CLI includes a complete manual prediction input."""
    provided_values = [args.exam_score is not None, args.years_exp is not None]
    if any(provided_values) and not all(provided_values):
        raise ValueError("--exam-score and --years-exp must be provided together.")
    return all(provided_values)


def build_prediction_features(
    *,
    exam_score: float,
    years_exp: float,
    preprocessor: Preprocessor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw and model-ready feature frames for one manual prediction."""
    raw_features = pd.DataFrame(
        [{"exam_score": exam_score, "years_exp": years_exp}],
        columns=preprocessor.feature_columns,
    )
    if preprocessor.scaler is None:
        return raw_features, raw_features

    model_features = pd.DataFrame(
        preprocessor.scaler.transform(raw_features),
        columns=preprocessor.feature_columns,
    )
    return raw_features, model_features


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        manual_prediction = has_prediction_input(args)
        data = DataLoader(args.data).load_data()
        preprocessor = Preprocessor(
            scaling=args.scaling,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        X_train, X_test, y_train, y_test = preprocessor.preprocess(data)

        trainer = Trainer(model_name=args.model, alpha=args.alpha)
        model = trainer.train(X_train, y_train)

        predictor = Predictor(model)
        predictions = predictor.predict(X_test)
        metrics = predictor.evaluate(y_test, predictions)
        if manual_prediction:
            raw_features, model_features = build_prediction_features(
                exam_score=args.exam_score,
                years_exp=args.years_exp,
                preprocessor=preprocessor,
            )
            manual_predictions = predictor.predict(model_features)
            print(f"Predicted salary for supplied input: {manual_predictions[0]:.2f}")
            if args.explain:
                explanation = predictor.explain(
                    model_features,
                    background_data=X_train,
                    display_features=raw_features,
                )
                print("SHAP feature contributions for supplied input:")
                print(explanation.feature_contributions.to_string(index=False))
        elif args.explain:
            explanation = predictor.explain(X_test, background_data=X_train)
            print("SHAP feature contributions for prediction row 0:")
            print(explanation.feature_contributions.to_string(index=False))

        print(f"Model: {args.model}")
        if hasattr(model, "coef_"):
            print("Coefficients:")
            for feature, coefficient in zip(preprocessor.feature_columns, model.coef_):
                print(f"  {feature}: {coefficient:.4f}")
        if hasattr(model, "intercept_"):
            print(f"Intercept: {model.intercept_:.4f}")

        output_path = (
            Path(args.model_output)
            if args.model_output
            else default_model_output(args.model)
        )
        saved_path = trainer.save_artifact(
            output_path,
            scaler=preprocessor.scaler,
            feature_columns=preprocessor.feature_columns,
            target_column=preprocessor.target_column,
        )
        print(f"Model artifact path: {saved_path}")
        print(f"Final metrics — RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
