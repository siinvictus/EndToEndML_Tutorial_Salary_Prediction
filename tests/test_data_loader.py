import pandas as pd
import pytest
from src.dataloader import DataLoader


@pytest.fixture  #this is a sample that will be used across the different tests that why the 'fixure' part is added.
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "exam_score": [70, 80, 90],
        "years_exp": [2, 5, 8],
        "salary": [50000, 60000, 70000]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_data_returns_dataframe(sample_csv) -> None:
    loader = DataLoader(sample_csv)
    data = loader.load_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (3, 3)


def test_load_data_raises_for_missing_file()-> None:
    loader = DataLoader("nonexistent_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_raises_for_unsupported_extension(tmp_path) -> None:
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("some content")
    loader = DataLoader(bad_file)
    with pytest.raises(ValueError):
        loader.load_data()