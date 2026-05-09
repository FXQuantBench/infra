"""Shared constants and builder functions used across the test suite."""

import pandas as pd

# 2023-01-01 00:00:00 UTC expressed as Unix milliseconds.
BASE_MS: int = 1_672_531_200_000


def make_tick_df(
    ts_ms_list: list[int],
    bid_prices: list[float],
    bid_vols: list[float],
    ask_prices: list[float],
    ask_vols: list[float],
) -> pd.DataFrame:
    """Build a DataFrame that matches the shape dukascopy_python.fetch() returns.

    The real API always returns all four price/volume columns regardless of
    offer_side; only the relevant columns are consumed by ingest_historical.py.
    """
    idx = pd.DatetimeIndex(
        pd.to_datetime(ts_ms_list, unit="ms", utc=True),
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "bidPrice": bid_prices,
            "bidVolume": bid_vols,
            "askPrice": ask_prices,
            "askVolume": ask_vols,
        },
        index=idx,
    )
