"""Gap validation for daily GBPUSD tick Parquet shards.

Called by daily_ingest.yml after ingest_historical.py writes a local shard,
before the file is uploaded to HuggingFace.
"""

import sys
from datetime import datetime, timezone


def validate_gaps(parquet_path: str, target: str) -> None:
    """Raise SystemExit(1) if any gap > 10 min exists during 07:00–22:00 UTC.

    Parameters
    ----------
    parquet_path:
        Path to the Parquet shard to validate.
    target:
        Date of the shard in ``YYYY-MM-DD`` format.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["timestamp_utc"])
    ts = sorted(table.column("timestamp_utc").to_pylist())

    day_epoch_ms = (
        int(
            datetime.strptime(target, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        * 1000
    )
    window_start = day_epoch_ms + 7 * 3600 * 1000   # 07:00 UTC in ms
    window_end   = day_epoch_ms + 22 * 3600 * 1000  # 22:00 UTC in ms

    window = [t for t in ts if window_start <= t <= window_end]

    if len(window) < 2:
        print("Fewer than 2 ticks in 07:00–22:00 UTC window — skipping gap check")
        return

    diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
    max_gap_ms = max(diffs)

    if max_gap_ms > 600_000:
        print(
            f"ERROR: gap of {max_gap_ms / 60_000:.1f} min detected during "
            f"07:00–22:00 UTC (threshold: 10 min)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Gap validation passed. Max gap in window: {max_gap_ms / 60_000:.2f} min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate tick gaps in a Parquet shard"
    )
    parser.add_argument("parquet_path", help="Path to the Parquet file")
    parser.add_argument("target_date", help="Date of the shard (YYYY-MM-DD)")
    args = parser.parse_args()

    validate_gaps(args.parquet_path, args.target_date)
