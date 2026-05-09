"""Tests for validate.py — gap validation logic."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from validate import validate_gaps

# 2023-01-02 00:00:00 UTC = 1 672 617 600 seconds = 1_672_617_600_000 ms
TARGET_DATE = "2023-01-02"
DAY_EPOCH_MS = 1_672_617_600_000

WINDOW_START = DAY_EPOCH_MS + 7 * 3600 * 1000   # 07:00 UTC
WINDOW_END   = DAY_EPOCH_MS + 22 * 3600 * 1000  # 22:00 UTC


def _write_ts_parquet(path: Path, timestamps_ms: list[int]) -> None:
    """Write a minimal Parquet file containing only timestamp_utc."""
    table = pa.table({"timestamp_utc": pa.array(timestamps_ms, type=pa.int64())})
    pq.write_table(table, str(path))


class TestValidateGaps:
    def test_uniform_ticks_one_minute_apart_passes(self, tmp_path):
        ticks = list(range(WINDOW_START, WINDOW_END, 60_000))
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, ticks)
        validate_gaps(str(path), TARGET_DATE)  # must not raise

    def test_exactly_ten_minute_gap_passes(self, tmp_path):
        # 600_000 ms == 10 min is NOT over the threshold (threshold is > 600_000)
        ticks = [WINDOW_START, WINDOW_START + 600_000, WINDOW_START + 660_000]
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, ticks)
        validate_gaps(str(path), TARGET_DATE)  # must not raise

    def test_eleven_minute_gap_in_window_fails(self, tmp_path):
        # Dense ticks for the first hour, then an 11-min gap, then dense again.
        first_hour = list(range(WINDOW_START, WINDOW_START + 60 * 60_000, 60_000))
        after_gap = list(range(WINDOW_START + 71 * 60_000, WINDOW_END, 60_000))
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, first_hour + after_gap)

        with pytest.raises(SystemExit) as exc_info:
            validate_gaps(str(path), TARGET_DATE)
        assert exc_info.value.code == 1

    def test_gap_before_window_is_ignored(self, tmp_path):
        # 11-min gap is entirely before 07:00; inside the window ticks are dense.
        pre_window = [DAY_EPOCH_MS, DAY_EPOCH_MS + 71 * 60_000]  # gap before window
        in_window = list(range(WINDOW_START, WINDOW_END, 60_000))
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, pre_window + in_window)
        validate_gaps(str(path), TARGET_DATE)  # must not raise

    def test_gap_after_window_is_ignored(self, tmp_path):
        # 11-min gap is entirely after 22:00; inside the window ticks are dense.
        in_window = list(range(WINDOW_START, WINDOW_END, 60_000))
        post_window = [WINDOW_END + 60_000, WINDOW_END + 72 * 60_000]
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, in_window + post_window)
        validate_gaps(str(path), TARGET_DATE)  # must not raise

    def test_fewer_than_two_ticks_in_window_skips_check(self, tmp_path):
        # Only one tick in the window — no diffs to compute, should not raise.
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, [WINDOW_START])
        validate_gaps(str(path), TARGET_DATE)  # must not raise

    def test_no_ticks_at_all_skips_check(self, tmp_path):
        # Ticks exist but all are outside the window.
        path = tmp_path / "ticks.parquet"
        _write_ts_parquet(path, [DAY_EPOCH_MS])  # before 07:00
        validate_gaps(str(path), TARGET_DATE)  # must not raise
