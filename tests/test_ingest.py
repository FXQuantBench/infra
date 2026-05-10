"""Tests for ingest_historical.py."""

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ingest_historical import (
    _load_uploaded_paths,
    _save_raw,
    _to_ms,
    default_end,
    fetch_ticks,
    main,
    parse_args,
    sha256_file,
    upload_and_update_manifest,
    upload_month_batch,
    write_parquet,
)
from tests.helpers import BASE_MS, make_tick_df


# ---------------------------------------------------------------------------
# _to_ms
# ---------------------------------------------------------------------------


class TestToMs:
    def test_naive_series_localized_to_utc(self):
        # 1970-01-01 00:00:01 naive → 1000 ms
        ts = pd.Series([pd.Timestamp("1970-01-01 00:00:01")])
        assert _to_ms(ts).iloc[0] == 1_000

    def test_tz_aware_series_converts_correctly(self):
        ts = pd.Series([pd.Timestamp(BASE_MS, unit="ms", tz="UTC")])
        assert _to_ms(ts).iloc[0] == BASE_MS

    def test_multiple_timestamps_all_converted(self):
        ts = pd.Series(
            pd.to_datetime([BASE_MS, BASE_MS + 500, BASE_MS + 1000], unit="ms", utc=True)
        )
        result = _to_ms(ts).tolist()
        assert result == [BASE_MS, BASE_MS + 500, BASE_MS + 1000]


# ---------------------------------------------------------------------------
# default_end
# ---------------------------------------------------------------------------


class TestDefaultEnd:
    def test_returns_last_day_of_previous_month(self):
        today = date.today()
        expected = date(today.year, today.month, 1) - timedelta(days=1)
        assert default_end() == expected


# ---------------------------------------------------------------------------
# fetch_ticks — merge logic
# ---------------------------------------------------------------------------


class TestFetchTicks:
    @patch("ingest_historical.dukascopy_python.fetch")
    def test_returns_none_when_bid_empty(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame()
        assert fetch_ticks(date(2023, 1, 1)) is None
        assert mock_fetch.call_count == 1

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_returns_none_when_ask_empty(self, mock_fetch):
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [1.30], [1.0])
        mock_fetch.side_effect = [bid_df, pd.DataFrame()]
        assert fetch_ticks(date(2023, 1, 1)) is None

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_merge_within_tolerance_not_interpolated(self, mock_fetch):
        """Bid and ask within 500 ms → is_interpolated=False for all rows."""
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS + 300], [0.0], [0.0], [1.3002], [0.8])
        mock_fetch.side_effect = [bid_df, ask_df]

        result = fetch_ticks(date(2023, 1, 1))

        assert result is not None
        assert len(result) == 1
        assert not result["is_interpolated"].iloc[0]
        assert result["ask"].iloc[0] == pytest.approx(1.3002)

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_ffill_beyond_tolerance_marks_interpolated(self, mock_fetch):
        """Ask > 500 ms away from a bid tick → ffill → is_interpolated=True."""
        # Two bid ticks; ask only at BASE_MS (matched); second bid is 1000ms away.
        bid_df = make_tick_df(
            [BASE_MS, BASE_MS + 1_000],
            [1.30, 1.31], [1.0, 1.0],
            [0.0, 0.0], [0.0, 0.0],
        )
        ask_df = make_tick_df([BASE_MS], [0.0], [0.0], [1.3002], [0.9])
        mock_fetch.side_effect = [bid_df, ask_df]

        result = fetch_ticks(date(2023, 1, 1))

        assert result is not None
        assert len(result) == 2
        assert not result["is_interpolated"].iloc[0]   # exact match
        assert result["is_interpolated"].iloc[1]        # forward-filled
        # Forward-filled value comes from the matched ask at BASE_MS
        assert result["ask"].iloc[1] == pytest.approx(1.3002)

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_drops_rows_with_no_prior_ask(self, mock_fetch):
        """Bid at t=0 with ask only at t+1000ms (> tolerance, no prior) → None."""
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS + 1_000], [0.0], [0.0], [1.3002], [0.9])
        mock_fetch.side_effect = [bid_df, ask_df]

        assert fetch_ticks(date(2023, 1, 1)) is None

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_schema_column_dtypes_are_correct(self, mock_fetch):
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS + 100], [0.0], [0.0], [1.3002], [0.8])
        mock_fetch.side_effect = [bid_df, ask_df]

        result = fetch_ticks(date(2023, 1, 1))

        assert result is not None
        assert result["timestamp_utc"].dtype == "int64"
        assert result["bid"].dtype == "float64"
        assert result["ask"].dtype == "float64"
        assert result["bid_volume"].dtype == "float32"
        assert result["ask_volume"].dtype == "float32"
        assert result["is_interpolated"].dtype == bool

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_output_columns_match_schema_exactly(self, mock_fetch):
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS], [0.0], [0.0], [1.3002], [0.8])
        mock_fetch.side_effect = [bid_df, ask_df]

        result = fetch_ticks(date(2023, 1, 1))

        assert list(result.columns) == [
            "timestamp_utc", "bid", "ask", "bid_volume", "ask_volume", "is_interpolated"
        ]

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_raw_dir_saves_bid_and_ask_parquets(self, mock_fetch, tmp_path):
        """With raw_dir set, raw bid and ask files must be written before merge."""
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS + 100], [0.0], [0.0], [1.3002], [0.8])
        mock_fetch.side_effect = [bid_df, ask_df]

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        fetch_ticks(date(2023, 1, 1), raw_dir=raw_dir)

        assert (raw_dir / "GBPUSD_bid_2023-01-01.parquet").exists()
        assert (raw_dir / "GBPUSD_ask_2023-01-01.parquet").exists()

    @patch("ingest_historical.dukascopy_python.fetch")
    def test_raw_parquet_contains_original_columns(self, mock_fetch, tmp_path):
        bid_df = make_tick_df([BASE_MS], [1.30], [1.0], [0.0], [0.0])
        ask_df = make_tick_df([BASE_MS + 100], [0.0], [0.0], [1.3002], [0.8])
        mock_fetch.side_effect = [bid_df, ask_df]

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        fetch_ticks(date(2023, 1, 1), raw_dir=raw_dir)

        import pyarrow.parquet as pq
        raw_bid = pq.read_table(str(raw_dir / "GBPUSD_bid_2023-01-01.parquet")).to_pandas()
        assert "bidPrice" in raw_bid.columns
        assert "bidVolume" in raw_bid.columns


# ---------------------------------------------------------------------------
# write_parquet
# ---------------------------------------------------------------------------


class TestWriteParquet:
    def test_roundtrip_preserves_values(self, sample_df, tmp_path):
        path = tmp_path / "out.parquet"
        write_parquet(sample_df, path)
        result = pq.read_table(str(path)).to_pandas()

        assert result["timestamp_utc"].iloc[0] == BASE_MS
        assert result["bid"].iloc[0] == pytest.approx(1.3000)
        assert result["ask"].iloc[0] == pytest.approx(1.3002)
        assert not result["is_interpolated"].iloc[0]

    def test_output_columns_match_schema(self, sample_df, tmp_path):
        path = tmp_path / "out.parquet"
        write_parquet(sample_df, path)
        result = pq.read_table(str(path)).to_pandas()

        assert list(result.columns) == [
            "timestamp_utc", "bid", "ask", "bid_volume", "ask_volume", "is_interpolated"
        ]

    def test_compression_is_zstd(self, sample_df, tmp_path):
        path = tmp_path / "out.parquet"
        write_parquet(sample_df, path)
        meta = pq.read_metadata(str(path))
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                assert rg.column(col_idx).compression == "ZSTD"


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


class TestSha256File:
    def test_known_content_produces_known_hash(self, tmp_path):
        content = b"hello, world"
        path = tmp_path / "test.bin"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert sha256_file(path) == expected

    def test_different_content_produces_different_hash(self, tmp_path):
        path_a = tmp_path / "a.bin"
        path_b = tmp_path / "b.bin"
        path_a.write_bytes(b"aaa")
        path_b.write_bytes(b"bbb")
        assert sha256_file(path_a) != sha256_file(path_b)


# ---------------------------------------------------------------------------
# upload_and_update_manifest
# ---------------------------------------------------------------------------


class TestUploadAndUpdateManifest:
    def test_new_entry_added_when_no_existing_manifest(self, tmp_parquet):
        api = MagicMock()
        api.hf_hub_download.side_effect = Exception("not found")

        upload_and_update_manifest(api, tmp_parquet, date(2023, 1, 1))

        assert api.upload_file.call_count == 2
        manifest_bytes = api.upload_file.call_args_list[1].kwargs["path_or_fileobj"]
        manifest = json.loads(manifest_bytes.decode())

        assert len(manifest["files"]) == 1
        entry = manifest["files"][0]
        assert entry["path"] == "GBPUSD/2023/01/01/ticks_2023-01-01.parquet"
        assert entry["rows"] == 1
        assert len(entry["sha256"]) == 64
        assert entry["date"] == "2023-01-01"
        assert manifest["last_updated"] is not None

    def test_existing_entry_replaced_not_duplicated(self, tmp_parquet, tmp_path):
        existing = {
            "files": [
                {
                    "path": "GBPUSD/2023/01/01/ticks_2023-01-01.parquet",
                    "rows": 999,
                    "sha256": "old_hash",
                    "date": "2023-01-01",
                },
                {
                    "path": "GBPUSD/2023/01/02/ticks_2023-01-02.parquet",
                    "rows": 100,
                    "sha256": "other_hash",
                    "date": "2023-01-02",
                },
            ],
            "last_updated": "2023-01-01T00:00:00+00:00",
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(existing))

        api = MagicMock()
        api.hf_hub_download.return_value = str(manifest_file)

        upload_and_update_manifest(api, tmp_parquet, date(2023, 1, 1))

        manifest_bytes = api.upload_file.call_args_list[1].kwargs["path_or_fileobj"]
        manifest = json.loads(manifest_bytes.decode())

        assert len(manifest["files"]) == 2
        paths = {e["path"] for e in manifest["files"]}
        assert "GBPUSD/2023/01/01/ticks_2023-01-01.parquet" in paths
        assert "GBPUSD/2023/01/02/ticks_2023-01-02.parquet" in paths

        updated = next(e for e in manifest["files"] if "01/01" in e["path"])
        assert updated["rows"] != 999
        assert updated["sha256"] != "old_hash"

    def test_parquet_uploaded_to_correct_hf_path(self, tmp_parquet):
        api = MagicMock()
        api.hf_hub_download.side_effect = Exception("not found")

        upload_and_update_manifest(api, tmp_parquet, date(2023, 1, 1))

        parquet_call = api.upload_file.call_args_list[0]
        assert parquet_call.kwargs["path_in_repo"] == "GBPUSD/2023/01/01/ticks_2023-01-01.parquet"
        assert parquet_call.kwargs["repo_id"] == "FXQuantBench/fx-ticks"
        assert parquet_call.kwargs["repo_type"] == "dataset"

    def test_manifest_uploaded_to_manifest_json(self, tmp_parquet):
        api = MagicMock()
        api.hf_hub_download.side_effect = Exception("not found")

        upload_and_update_manifest(api, tmp_parquet, date(2023, 1, 1))

        manifest_call = api.upload_file.call_args_list[1]
        assert manifest_call.kwargs["path_in_repo"] == "manifest.json"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        with patch.object(sys, "argv", ["ingest_historical.py"]):
            args = parse_args()
        assert args.start == "2020-01-01"
        assert args.end is None
        assert args.output_dir is None

    def test_explicit_values(self):
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-31",
             "--output-dir", "/tmp/ticks", "--raw-dir", "/tmp/raw"],
        ):
            args = parse_args()
        assert args.start == "2023-01-01"
        assert args.end == "2023-01-31"
        assert args.output_dir == "/tmp/ticks"
        assert args.raw_dir == "/tmp/raw"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    @patch("ingest_historical.fetch_ticks")
    def test_local_mode_writes_parquet(self, mock_fetch, sample_df, tmp_path):
        mock_fetch.return_value = sample_df
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01",
             "--output-dir", str(tmp_path)],
        ):
            main()

        assert (tmp_path / "ticks_2023-01-01.parquet").exists()

    @patch("ingest_historical.fetch_ticks")
    def test_local_mode_skips_day_with_no_data(self, mock_fetch, tmp_path):
        mock_fetch.return_value = None
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01",
             "--output-dir", str(tmp_path)],
        ):
            main()

        assert not list(tmp_path.glob("*.parquet"))

    @patch("ingest_historical.fetch_ticks")
    def test_local_mode_continues_after_fetch_exception(self, mock_fetch, tmp_path):
        mock_fetch.side_effect = [RuntimeError("network error"), None]
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-02",
             "--output-dir", str(tmp_path)],
        ):
            main()  # should not raise

    @patch("ingest_historical.upload_month_batch")
    @patch("ingest_historical.HfApi")
    @patch("ingest_historical.fetch_ticks")
    def test_upload_mode_calls_upload_month_batch(
        self, mock_fetch, mock_hf_cls, mock_upload_batch, sample_df
    ):
        mock_fetch.return_value = sample_df
        mock_api = MagicMock()
        mock_hf_cls.return_value = mock_api

        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01"],
        ):
            main()

        mock_upload_batch.assert_called_once()
        call_args = mock_upload_batch.call_args
        assert call_args.args[0] is mock_api
        # Second arg is a list of (date, path) tuples for the month.
        month_files = call_args.args[1]
        assert len(month_files) == 1
        assert month_files[0][0] == date(2023, 1, 1)

    def test_start_after_end_raises_system_exit(self):
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-02-01", "--end", "2023-01-01"],
        ):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# _load_uploaded_paths
# ---------------------------------------------------------------------------


class TestLoadUploadedPaths:
    def test_returns_paths_from_manifest(self, tmp_path):
        manifest = {
            "files": [
                {"path": "GBPUSD/2023/01/01/ticks_2023-01-01.parquet"},
                {"path": "GBPUSD/2023/01/02/ticks_2023-01-02.parquet"},
            ],
            "last_updated": "2023-01-02T00:00:00+00:00",
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        api = MagicMock()
        api.hf_hub_download.return_value = str(manifest_file)

        result = _load_uploaded_paths(api)
        assert result == {
            "GBPUSD/2023/01/01/ticks_2023-01-01.parquet",
            "GBPUSD/2023/01/02/ticks_2023-01-02.parquet",
        }

    def test_returns_empty_set_when_manifest_missing(self):
        api = MagicMock()
        api.hf_hub_download.side_effect = Exception("not found")
        assert _load_uploaded_paths(api) == set()

    def test_returns_empty_set_when_files_list_empty(self, tmp_path):
        manifest = {"files": [], "last_updated": None}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        api = MagicMock()
        api.hf_hub_download.return_value = str(manifest_file)
        assert _load_uploaded_paths(api) == set()


# ---------------------------------------------------------------------------
# Resume logic via main()
# ---------------------------------------------------------------------------


class TestResume:
    @patch("ingest_historical.fetch_ticks")
    def test_local_mode_skips_existing_file(self, mock_fetch, sample_df, tmp_path):
        # Pre-write the file so it looks like a previous run succeeded.
        existing = tmp_path / "ticks_2023-01-01.parquet"
        write_parquet(sample_df, existing)

        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01",
             "--output-dir", str(tmp_path)],
        ):
            main()

        # fetch_ticks must never be called for days already on disk.
        mock_fetch.assert_not_called()

    @patch("ingest_historical.fetch_ticks")
    def test_local_mode_processes_missing_day(self, mock_fetch, sample_df, tmp_path):
        mock_fetch.return_value = sample_df
        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01",
             "--output-dir", str(tmp_path)],
        ):
            main()

        mock_fetch.assert_called_once()
        assert (tmp_path / "ticks_2023-01-01.parquet").exists()

    @patch("ingest_historical.upload_month_batch")
    @patch("ingest_historical.HfApi")
    @patch("ingest_historical.fetch_ticks")
    def test_upload_mode_skips_day_in_manifest(
        self, mock_fetch, mock_hf_cls, mock_upload_batch, tmp_path
    ):
        manifest = {
            "files": [{"path": "GBPUSD/2023/01/01/ticks_2023-01-01.parquet"}],
            "last_updated": "2023-01-01T00:00:00+00:00",
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        mock_api = MagicMock()
        mock_hf_cls.return_value = mock_api
        mock_api.hf_hub_download.return_value = str(manifest_file)

        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01"],
        ):
            main()

        mock_fetch.assert_not_called()
        mock_upload_batch.assert_not_called()

    @patch("ingest_historical.upload_month_batch")
    @patch("ingest_historical.HfApi")
    @patch("ingest_historical.fetch_ticks")
    def test_upload_mode_processes_day_not_in_manifest(
        self, mock_fetch, mock_hf_cls, mock_upload_batch, sample_df, tmp_path
    ):
        manifest = {"files": [], "last_updated": None}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        mock_api = MagicMock()
        mock_hf_cls.return_value = mock_api
        mock_api.hf_hub_download.return_value = str(manifest_file)
        mock_fetch.return_value = sample_df

        with patch.object(
            sys, "argv",
            ["ingest_historical.py", "--start", "2023-01-01", "--end", "2023-01-01"],
        ):
            main()

        mock_fetch.assert_called_once()
        mock_upload_batch.assert_called_once()

