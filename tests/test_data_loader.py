import pandas as pd
import pytest

from src.dataloader import DataLoader


def test_load_data_reads_csv_file(tmp_path, salary_dataframe: pd.DataFrame) -> None:
    csv_path = tmp_path / "salary_data.csv"
    salary_dataframe.to_csv(csv_path, index=False)

    data = DataLoader(csv_path).load_data()

    pd.testing.assert_frame_equal(data, salary_dataframe)


def test_load_data_reads_excel_file(tmp_path, salary_dataframe: pd.DataFrame) -> None:
    excel_path = tmp_path / "salary_data.xlsx"
    salary_dataframe.to_excel(excel_path, index=False)

    data = DataLoader(excel_path).load_data()

    pd.testing.assert_frame_equal(data, salary_dataframe, check_dtype=False)


def test_load_data_accepts_string_paths(tmp_path, salary_dataframe: pd.DataFrame) -> None:
    csv_path = tmp_path / "salary_data.csv"
    salary_dataframe.to_csv(csv_path, index=False)

    data = DataLoader(str(csv_path)).load_data()

    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["exam_score", "years_exp", "salary"]


def test_load_data_raises_for_missing_file() -> None:
    loader = DataLoader("nonexistent_file.csv")

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        loader.load_data()


def test_load_data_raises_for_unsupported_extension(tmp_path) -> None:
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("some content")
    loader = DataLoader(bad_file)

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_data()
