import pandas as pd
import pytest

from src.dataloader import DataLoader


def test_load_csv_returns_dataframe(tmp_path):
    data_path = tmp_path / "salary.csv"
    pd.DataFrame({"salary": [50_000], "years_exp": [2]}).to_csv(data_path, index=False)

    loaded_data = DataLoader(data_path).load_data()

    assert list(loaded_data.columns) == ["salary", "years_exp"]
    assert loaded_data.shape == (1, 2)


def test_load_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        DataLoader(tmp_path / "missing.csv").load_data()


def test_load_data_rejects_unsupported_file_type(tmp_path):
    data_path = tmp_path / "salary.json"
    data_path.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DataLoader(data_path).load_data()
