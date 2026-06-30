import argparse
from pathlib import Path

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
        help="Print SHAP feature importance for the test set.",
    )
    return parser


def default_model_output(model_name: str) -> Path:
    """Return the standard artifact path for a model type."""
    filename = "linear_regression.pkl" if model_name == "linear" else "lasso_regression.pkl"
    return Path("models") / filename


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
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
        if args.explain:
            explanation = predictor.explain(X_test, background_data=X_train)
            print("SHAP feature importance:")
            print(explanation.feature_importance.to_string(index=False))

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
