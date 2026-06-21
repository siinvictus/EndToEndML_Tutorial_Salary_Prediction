import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    """Clean salary data, select features, split it, and optionally scale it."""

    COLUMN_RENAMES = {
        "#": "index",
        "Exam Score (0–100)": "exam_score",
        "Exam Score (0â€“100)": "exam_score",
        "Years of Experience": "years_exp",
        "Salary (€)": "salary",
        "Salary (â‚¬)": "salary",
    }
    SCALING_OPTIONS = {"standard", "minmax", "none"}

    def __init__(
        self,
        target_column: str = "salary",
        scaling: str = "standard",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if scaling not in self.SCALING_OPTIONS:
            options = ", ".join(sorted(self.SCALING_OPTIONS))
            raise ValueError(f"Unsupported scaling '{scaling}'. Choose one of: {options}.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")

        self.target_column = target_column
        self.scaling = scaling
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns: list[str] = []
        self.scaler: StandardScaler | MinMaxScaler | None = None

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename known notebook column names to stable snake_case names."""
        renamed_data = data.rename(columns=self.COLUMN_RENAMES).copy()
        print(f"Columns after renaming: {', '.join(map(str, renamed_data.columns))}")
        return renamed_data

    def drop_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values and report how many were removed."""
        cleaned_data = data.dropna().copy()
        dropped_rows = len(data) - len(cleaned_data)
        print(f"Dropped {dropped_rows} row(s) with missing values.")
        return cleaned_data

    def split_features_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return numeric features and the configured target column."""
        if self.target_column not in data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' was not found. "
                f"Available columns: {', '.join(map(str, data.columns))}"
            )
        if not pd.api.types.is_numeric_dtype(data[self.target_column]):
            raise ValueError(f"Target column '{self.target_column}' must be numeric.")

        candidate_features = data.drop(
            columns=[self.target_column, "index"], errors="ignore"
        )
        features = candidate_features.select_dtypes(include=np.number).copy()
        if features.empty:
            raise ValueError("No numeric feature columns are available for training.")

        self.feature_columns = list(features.columns)
        target = data[self.target_column].copy()
        print(
            f"Features: {', '.join(self.feature_columns)}; "
            f"target: {self.target_column}"
        )
        return features, target

    def split_train_test(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create reproducible train and test datasets."""
        return train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit a scaler on training features only, then transform both splits."""
        if self.scaling == "none":
            self.scaler = None
            return X_train, X_test

        self.scaler = (
            StandardScaler() if self.scaling == "standard" else MinMaxScaler()
        )
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_columns,
            index=X_test.index,
        )
        print(f"Applied {self.scaling} scaling using training data only.")
        return X_train_scaled, X_test_scaled

    def preprocess(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run the reusable preprocessing workflow from raw data to model splits."""
        renamed_data = self.rename_columns(data)
        cleaned_data = self.drop_nulls(renamed_data)
        features, target = self.split_features_target(cleaned_data)
        X_train, X_test, y_train, y_test = self.split_train_test(features, target)
        X_train, X_test = self.scale_features(X_train, X_test)
        return X_train, X_test, y_train, y_test
