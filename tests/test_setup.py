"""Tests for setup_hf_dataset.py."""

import json
from unittest.mock import MagicMock, patch

from setup_hf_dataset import main


class TestSetupMain:
    @patch("setup_hf_dataset.HfApi")
    def test_uploads_two_files(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()
        assert mock_api.upload_file.call_count == 2

    @patch("setup_hf_dataset.HfApi")
    def test_uploads_readme_and_manifest(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()
        paths = {c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list}
        assert paths == {"README.md", "manifest.json"}

    @patch("setup_hf_dataset.HfApi")
    def test_readme_contains_all_schema_columns(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()

        calls = mock_api.upload_file.call_args_list
        readme_call = next(c for c in calls if c.kwargs["path_in_repo"] == "README.md")
        content = readme_call.kwargs["path_or_fileobj"].decode()

        for col in ("timestamp_utc", "bid", "ask", "bid_volume", "ask_volume", "is_interpolated"):
            assert col in content, f"Column '{col}' missing from dataset card"

    @patch("setup_hf_dataset.HfApi")
    def test_readme_contains_duckdb_hf_example(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()

        calls = mock_api.upload_file.call_args_list
        readme_call = next(c for c in calls if c.kwargs["path_in_repo"] == "README.md")
        content = readme_call.kwargs["path_or_fileobj"].decode()

        assert "read_parquet" in content
        assert "hf://" in content

    @patch("setup_hf_dataset.HfApi")
    def test_initial_manifest_is_empty(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()

        calls = mock_api.upload_file.call_args_list
        manifest_call = next(c for c in calls if c.kwargs["path_in_repo"] == "manifest.json")
        manifest = json.loads(manifest_call.kwargs["path_or_fileobj"].decode())

        assert manifest["files"] == []
        assert manifest["last_updated"] is None

    @patch("setup_hf_dataset.HfApi")
    def test_targets_correct_repo(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        main()

        for call in mock_api.upload_file.call_args_list:
            assert call.kwargs["repo_id"] == "FXQuantBench/fx-ticks"
            assert call.kwargs["repo_type"] == "dataset"
