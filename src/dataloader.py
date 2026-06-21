from pathlib import Path

import pandas as pd


class DataLoader:

    SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        extension = self.data_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{extension or 'none'}'. "
                f"Supported file types: {supported}."
            )

        try:
            if extension == ".csv":
                data = pd.read_csv(self.data_path)
            else:
                data = pd.read_excel(self.data_path)
        except ImportError as error:
            raise ValueError(
                f"Cannot load '{extension}' files because an Excel reader is missing. "
                "Install the project dependencies with 'uv sync'."
            ) from error

        print(f"Loaded dataset: {data.shape[0]} rows x {data.shape[1]} columns")
        print(f"Columns: {', '.join(map(str, data.columns))}")
        return data
