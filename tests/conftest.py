"""Shared pytest fixtures."""

import pandas as pd
import pytest

from ingest_historical import write_parquet
from tests.helpers import BASE_MS


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """One valid row matching the full output schema of ingest_historical."""
    return pd.DataFrame(
        {
            "timestamp_utc": pd.array([BASE_MS], dtype="int64"),
            "bid": pd.array([1.3000], dtype="float64"),
            "ask": pd.array([1.3002], dtype="float64"),
            "bid_volume": pd.array([1.0], dtype="float32"),
            "ask_volume": pd.array([0.8], dtype="float32"),
            "is_interpolated": [False],
        }
    )


@pytest.fixture()
def tmp_parquet(tmp_path, sample_df) -> "pathlib.Path":
    """Write sample_df to a temporary Parquet file and return its path."""
    path = tmp_path / "ticks_2023-01-01.parquet"
    write_parquet(sample_df, path)
    return path
