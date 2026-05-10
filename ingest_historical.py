"""Ingest historical GBPUSD tick data from Dukascopy into HuggingFace.

Usage
-----
# Bulk historical (uploads to HF + updates manifest):
    python ingest_historical.py --start 2020-01-01 --end 2023-12-31

# Local-only (for workflow: write Parquet, skip HF upload):
    python ingest_historical.py --start 2024-05-08 --end 2024-05-08 --output-dir /tmp/ticks

# Save raw bid/ask before merge (for debugging merge logic):
    python ingest_historical.py --start 2023-01-01 --end 2023-01-01 \
        --output-dir /tmp/ticks --raw-dir /tmp/raw
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import dukascopy_python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD
from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "FXQuantBench/fx-ticks"
REPO_TYPE = "dataset"

SCHEMA = pa.schema(
    [
        pa.field("timestamp_utc", pa.int64()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_volume", pa.float32()),
        pa.field("ask_volume", pa.float32()),
        pa.field("is_interpolated", pa.bool_()),
    ]
)

ROW_GROUP_SIZE = 50_000
MERGE_TOLERANCE_MS = 500


def default_end() -> date:
    """Last calendar day of the most recently completed month."""
    today = date.today()
    return date(today.year, today.month, 1) - timedelta(days=1)


def _to_ms(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to int64 Unix milliseconds (assumes UTC)."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    return series.astype("int64") // 1_000_000


def _save_raw(df: pd.DataFrame, raw_dir: Path, side: str, day: date) -> None:
    """Persist a raw dukascopy DataFrame (pre-merge) as Parquet."""
    path = raw_dir / f"GBPUSD_{side}_{day.isoformat()}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="zstd")
    print(f"  Raw {side} saved → {path}")


def fetch_ticks(
    day: date, raw_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """Download and merge bid/ask ticks for *day*.

    Parameters
    ----------
    day:
        Calendar day to fetch.
    raw_dir:
        If given, write the raw (pre-merge) bid and ask DataFrames as Parquet
        to this directory before any normalisation or merging.

    Returns a DataFrame with schema columns or None if no data.
    """
    start = datetime(day.year, day.month, day.day)
    end = datetime(day.year, day.month, day.day) + timedelta(days=1)

    df_bid = dukascopy_python.fetch(
        INSTRUMENT_FX_MAJORS_GBP_USD,
        dukascopy_python.INTERVAL_TICK,
        dukascopy_python.OFFER_SIDE_BID,
        start=start,
        end=end,
    )
    if df_bid is None or df_bid.empty:
        return None

    df_ask = dukascopy_python.fetch(
        INSTRUMENT_FX_MAJORS_GBP_USD,
        dukascopy_python.INTERVAL_TICK,
        dukascopy_python.OFFER_SIDE_ASK,
        start=start,
        end=end,
    )
    if df_ask is None or df_ask.empty:
        return None

    # --- Optionally persist raw data before any transformation ---
    if raw_dir is not None:
        _save_raw(df_bid, raw_dir, "bid", day)
        _save_raw(df_ask, raw_dir, "ask", day)

    # --- Normalise bid side ---
    df_bid = df_bid.reset_index()
    df_bid["timestamp_utc"] = _to_ms(df_bid["timestamp"])
    df_bid = (
        df_bid[["timestamp_utc", "bidPrice", "bidVolume"]]
        .rename(columns={"bidPrice": "bid", "bidVolume": "bid_volume"})
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # --- Normalise ask side ---
    df_ask = df_ask.reset_index()
    df_ask["timestamp_utc"] = _to_ms(df_ask["timestamp"])
    df_ask = (
        df_ask[["timestamp_utc", "askPrice", "askVolume"]]
        .rename(columns={"askPrice": "ask", "askVolume": "ask_volume"})
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # --- Nearest-tick merge (bid is primary, 500 ms tolerance) ---
    merged = pd.merge_asof(
        df_bid,
        df_ask,
        on="timestamp_utc",
        direction="nearest",
        tolerance=MERGE_TOLERANCE_MS,
    )

    # Forward-fill ask where no tick was found within tolerance; mark rows.
    merged["is_interpolated"] = False
    ask_missing = merged["ask"].isna()
    if ask_missing.any():
        merged[["ask", "ask_volume"]] = merged[["ask", "ask_volume"]].ffill()
        # Only rows that were missing AND now have a value (ffill succeeded)
        newly_filled = ask_missing & merged["ask"].notna()
        merged.loc[newly_filled, "is_interpolated"] = True

    # Drop rows where ask is still NaN (no prior data — start-of-session).
    merged = merged.dropna(subset=["ask"])

    if merged.empty:
        return None

    # --- Enforce schema types ---
    merged["timestamp_utc"] = merged["timestamp_utc"].astype("int64")
    merged["bid"] = merged["bid"].astype("float64")
    merged["ask"] = merged["ask"].astype("float64")
    merged["bid_volume"] = merged["bid_volume"].astype("float32")
    merged["ask_volume"] = merged["ask_volume"].astype("float32")
    merged["is_interpolated"] = merged["is_interpolated"].astype(bool)

    return merged[
        ["timestamp_utc", "bid", "ask", "bid_volume", "ask_volume", "is_interpolated"]
    ]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        compression="zstd",
        row_group_size=ROW_GROUP_SIZE,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_upload(api: HfApi, **kwargs) -> None:
    """Upload a file to HF with exponential backoff on 429 rate-limit errors."""
    import requests

    delay = 10
    for attempt in range(6):
        try:
            api.upload_file(**kwargs)
            return
        except Exception as exc:
            is_429 = (
                isinstance(exc, requests.exceptions.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 429
            ) or "429" in str(exc)
            if is_429 and attempt < 5:
                print(f"  HF rate-limited, retrying in {delay}s …")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                raise


def upload_month_batch(api: HfApi, month_files: list[tuple[date, Path]]) -> None:
    """Upload all parquet files for a month plus an updated manifest in one commit.

    Parameters
    ----------
    api:
        Authenticated HfApi instance.
    month_files:
        Ordered list of (day, tmp_path) pairs whose data should be committed.
    """
    import requests

    # Load current manifest.
    try:
        manifest_dl = api.hf_hub_download(
            repo_id=REPO_ID,
            filename="manifest.json",
            repo_type=REPO_TYPE,
        )
        with open(manifest_dl) as f:
            manifest = json.load(f)
    except Exception:
        manifest = {"files": [], "last_updated": None}

    operations: list[CommitOperationAdd] = []
    for day, tmp_path in month_files:
        hf_path = (
            f"GBPUSD/{day.year}/{day.month:02d}/{day.day:02d}"
            f"/ticks_{day.isoformat()}.parquet"
        )
        checksum = sha256_file(tmp_path)
        row_count = pq.read_metadata(str(tmp_path)).num_rows

        operations.append(
            CommitOperationAdd(
                path_in_repo=hf_path,
                path_or_fileobj=str(tmp_path),
            )
        )
        manifest["files"] = [e for e in manifest["files"] if e.get("path") != hf_path]
        manifest["files"].append(
            {
                "path": hf_path,
                "rows": row_count,
                "sha256": checksum,
                "date": day.isoformat(),
            }
        )

    manifest["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
    operations.append(
        CommitOperationAdd(
            path_in_repo="manifest.json",
            path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        )
    )

    month_label = month_files[0][0].strftime("%Y-%m")
    delay = 10
    for attempt in range(6):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=(
                    f"Add {month_label} ticks ({len(month_files)} days)"
                ),
            )
            print(
                f"  Uploaded {len(month_files)} day(s) for {month_label} + manifest"
            )
            return
        except Exception as exc:
            is_429 = (
                isinstance(exc, requests.exceptions.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 429
            ) or "429" in str(exc)
            if is_429 and attempt < 5:
                print(f"  HF rate-limited, retrying in {delay}s …")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                raise


def _flush_month(api: HfApi, pending: list[tuple[date, Path]]) -> None:
    """Commit *pending* files as a single HF commit then clean up staged/temp files.

    Files are deleted only after a successful commit so that a crash or rate-limit
    failure leaves them on disk for the next run to resume from.
    """
    if not pending:
        return
    upload_month_batch(api, pending)  # raises on failure; files survive for resume
    for _, p in pending:
        p.unlink(missing_ok=True)


def upload_and_update_manifest(api: HfApi, local_path: Path, day: date) -> None:
    hf_path = (
        f"GBPUSD/{day.year}/{day.month:02d}/{day.day:02d}"
        f"/ticks_{day.isoformat()}.parquet"
    )

    _hf_upload(
        api,
        path_or_fileobj=str(local_path),
        path_in_repo=hf_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"  Uploaded {hf_path}")

    checksum = sha256_file(local_path)
    row_count = pq.read_metadata(str(local_path)).num_rows

    # Download current manifest, update, re-upload.
    try:
        manifest_path = api.hf_hub_download(
            repo_id=REPO_ID,
            filename="manifest.json",
            repo_type=REPO_TYPE,
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:
        manifest = {"files": [], "last_updated": None}

    # Replace existing entry for the same HF path, then append updated entry.
    manifest["files"] = [e for e in manifest["files"] if e.get("path") != hf_path]
    manifest["files"].append(
        {
            "path": hf_path,
            "rows": row_count,
            "sha256": checksum,
            "date": day.isoformat(),
        }
    )
    manifest["last_updated"] = datetime.now(tz=timezone.utc).isoformat()

    _hf_upload(
        api,
        path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        path_in_repo="manifest.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print("  Updated manifest.json")


def _load_uploaded_paths(api: HfApi) -> set[str]:
    """Return the set of HF paths already recorded in manifest.json.

    Used by upload mode to skip days that were successfully uploaded in a
    previous run, allowing resume after an unexpected failure.
    """
    try:
        manifest_path = api.hf_hub_download(
            repo_id=REPO_ID,
            filename="manifest.json",
            repo_type=REPO_TYPE,
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {e["path"] for e in manifest.get("files", [])}
    except Exception:
        return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest GBPUSD ticks from Dukascopy")
    parser.add_argument(
        "--start",
        default="2020-01-01",
        metavar="YYYY-MM-DD",
        help="First date to ingest (inclusive, default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last date to ingest (inclusive, default: last day of previous month)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help=(
            "Write Parquet files to this local directory and skip HF upload. "
            "Directory is created if it does not exist."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        metavar="PATH",
        help=(
            "Save raw (pre-merge) bid and ask Parquet files here for each day. "
            "Useful for debugging merge logic. Directory is created if it does not exist."
        ),
    )
    parser.add_argument(
        "--stage-dir",
        default=None,
        metavar="PATH",
        help=(
            "Persist merged Parquet files here between runs (upload mode only). "
            "Days already staged on disk skip Dukascopy re-fetch; files are deleted "
            "only after a successful HF commit. Directory is created if it does not exist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else default_end()

    if start > end:
        raise SystemExit(f"--start {start} is after --end {end}")

    local_mode = args.output_dir is not None
    api = None if local_mode else HfApi()

    if local_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir: Optional[Path] = None
    if args.raw_dir is not None:
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    stage_dir: Optional[Path] = None
    if not local_mode and args.stage_dir is not None:
        stage_dir = Path(args.stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume support ---
    # In local mode: skip days whose output file already exists on disk.
    # In upload mode: fetch the manifest once and skip days already uploaded.
    uploaded_paths: set[str] = set()
    if not local_mode:
        print("Fetching manifest to determine already-uploaded days ...")
        uploaded_paths = _load_uploaded_paths(api)
        if uploaded_paths:
            print(f"  {len(uploaded_paths)} day(s) already in manifest — will skip.")

    # Accumulate (day, path) pairs across the current month in upload mode.
    # Uploads run in a single background thread so Dukascopy fetching continues
    # while HF is resolving a rate-limit (429) back-off.
    pending_uploads: list[tuple[date, Path]] = []
    pending_month: Optional[tuple[int, int]] = None
    upload_futures: list[Future] = []
    upload_executor = (
        None if local_mode else ThreadPoolExecutor(max_workers=1)
    )

    current = start
    while current <= end:
        print(f"Processing {current.isoformat()} ...")

        if local_mode:
            # --- Local mode ---
            local_path = output_dir / f"ticks_{current.isoformat()}.parquet"
            if local_path.exists():
                print("  Already exists locally — skipping")
                current += timedelta(days=1)
                continue

            try:
                df = fetch_ticks(current, raw_dir=raw_dir)
            except Exception as exc:
                print(f"  WARNING: fetch failed — {exc}")
                current += timedelta(days=1)
                continue

            if df is None:
                print("  No data — skipping")
                current += timedelta(days=1)
                continue

            write_parquet(df, local_path)
            print(f"  Written to {local_path} ({len(df):,} rows)")

        else:
            # --- Upload mode ---
            hf_path = (
                f"GBPUSD/{current.year}/{current.month:02d}/{current.day:02d}"
                f"/ticks_{current.isoformat()}.parquet"
            )
            staged_path = (
                stage_dir / f"ticks_{current.isoformat()}.parquet"
                if stage_dir else None
            )

            if hf_path in uploaded_paths:
                print("  Already in manifest — skipping")
                # Remove any orphaned staged file left from a previous partial run.
                if staged_path and staged_path.exists():
                    staged_path.unlink()
                    print("  Removed orphaned staged file")
                current += timedelta(days=1)
                continue

            # Use staged file if available, otherwise fetch from Dukascopy.
            if staged_path and staged_path.exists():
                print("  Already staged — queuing for upload without re-fetch")
                tmp_path = staged_path
            else:
                try:
                    df = fetch_ticks(current, raw_dir=raw_dir)
                except Exception as exc:
                    print(f"  WARNING: fetch failed — {exc}")
                    current += timedelta(days=1)
                    continue

                if df is None:
                    print("  No data — skipping")
                    current += timedelta(days=1)
                    continue

                if staged_path:
                    write_parquet(df, staged_path)
                    tmp_path = staged_path
                    print(f"  Staged {len(df):,} rows → {staged_path.name}")
                else:
                    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet")
                    tmp_path = Path(tmp_name)
                    os.close(tmp_fd)
                    write_parquet(df, tmp_path)
                    print(f"  Fetched {len(df):,} rows")

            this_month = (current.year, current.month)
            if pending_month is not None and pending_month != this_month:
                # Month boundary — submit completed month for background upload.
                batch = list(pending_uploads)
                upload_futures.append(
                    upload_executor.submit(_flush_month, api, batch)
                )
                pending_uploads = []
            pending_month = this_month
            pending_uploads.append((current, tmp_path))

        current += timedelta(days=1)
        # Brief pause between days to avoid saturating Dukascopy's CDN.
        time.sleep(2)

    # Submit the final (possibly partial) month and wait for all uploads.
    if not local_mode:
        if pending_uploads:
            upload_futures.append(
                upload_executor.submit(_flush_month, api, pending_uploads)
            )
        upload_executor.shutdown(wait=True)
        # Re-raise any exception from upload threads.
        for f in upload_futures:
            f.result()


if __name__ == "__main__":
    main()
