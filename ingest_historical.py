"""Ingest historical GBPUSD tick data from Dukascopy into HuggingFace.

Usage
-----
# Bulk historical (uploads to HF + updates manifest):
    python ingest_historical.py --start 2020-01-01 --end 2023-12-31

# Local-only (for workflow: write Parquet, skip HF upload):
    python ingest_historical.py --start 2024-05-08 --end 2024-05-08 --output-dir /tmp/ticks
"""

import argparse
import hashlib
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import dukascopy_python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD
from huggingface_hub import HfApi

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


def fetch_ticks(day: date) -> Optional[pd.DataFrame]:
    """Download and merge bid/ask ticks for *day*.

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


def upload_and_update_manifest(api: HfApi, local_path: Path, day: date) -> None:
    hf_path = (
        f"GBPUSD/{day.year}/{day.month:02d}/{day.day:02d}"
        f"/ticks_{day.isoformat()}.parquet"
    )

    api.upload_file(
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

    api.upload_file(
        path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        path_in_repo="manifest.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print("  Updated manifest.json")


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

    current = start
    while current <= end:
        print(f"Processing {current.isoformat()} ...")
        try:
            df = fetch_ticks(current)
        except Exception as exc:
            print(f"  WARNING: fetch failed — {exc}")
            current += timedelta(days=1)
            continue

        if df is None:
            print("  No data — skipping")
            current += timedelta(days=1)
            continue

        if local_mode:
            local_path = output_dir / f"ticks_{current.isoformat()}.parquet"
            write_parquet(df, local_path)
            print(f"  Written to {local_path} ({len(df):,} rows)")
        else:
            # Write to a temp file, upload, then clean up.
            tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet")
            tmp_path = Path(tmp_name)
            try:
                import os
                os.close(tmp_fd)
                write_parquet(df, tmp_path)
                upload_and_update_manifest(api, tmp_path, current)
                print(f"  Done ({len(df):,} rows)")
            finally:
                tmp_path.unlink(missing_ok=True)

        current += timedelta(days=1)


if __name__ == "__main__":
    main()
