import pandas as pd
import pytest
from sklearn.utils import shuffle

from src.create_balance import read_and_balance


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "text": ["This is a sample text.", "Another sample text."],
            "category": [0, 1],
        }
    )


def test_pre_process(sample_data):
    # Test that pre_process returns the expected number of rows and columns
    X, Y, _ = read_and_balance.pre_process(sample_data)
    assert X.shape[0] == sample_data.shape[0]
    assert Y.shape[0] == sample_data.shape[0]

    # Test that pre_process drops null values and duplicates
    data_with_nulls = sample_data.copy()
    data_with_nulls.iloc[0, 0] = None
    data_with_duplicates = pd.concat(
        [sample_data, sample_data], ignore_index=True
    )
    X, Y, _ = read_and_balance.pre_process(data_with_nulls)
    assert X.shape[0] == sample_data.shape[0] - 1
    assert Y.shape[0] == sample_data.shape[0] - 1
    X, Y, _ = read_and_balance.pre_process(data_with_duplicates)
    assert X.shape[0] == sample_data.shape[0]
    assert Y.shape[0] == sample_data.shape[0]
