"""Bootstrap the FXQuantBench/fx-ticks HF dataset with a card and empty manifest."""

import json

from huggingface_hub import HfApi

REPO_ID = "FXQuantBench/fx-ticks"
REPO_TYPE = "dataset"

DATASET_CARD = """\
---
license: mit
task_categories: []
---
# FXQuantBench / fx-ticks

Tick-level GBPUSD bid/ask data ingested from Dukascopy.

## Column Schema

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_utc` | int64 | Unix ms, UTC, no nulls |
| `bid` | float64 | |
| `ask` | float64 | |
| `bid_volume` | float32 | |
| `ask_volume` | float32 | |
| `is_interpolated` | bool | True if either side was forward-filled within 500 ms |

## HF Path Convention

```
GBPUSD/{YYYY}/{MM}/{DD}/ticks_{YYYY-MM-DD}.parquet
```

Example: `GBPUSD/2023/01/01/ticks_2023-01-01.parquet`

## Example DuckDB Query

```sql
SELECT COUNT(*)
FROM read_parquet('hf://datasets/FXQuantBench/fx-ticks/GBPUSD/2023/01/01/ticks_2023-01-01.parquet')
```
"""

INITIAL_MANIFEST = {"files": [], "last_updated": None}


def main() -> None:
    api = HfApi()

    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print("Uploaded README.md (dataset card)")

    api.upload_file(
        path_or_fileobj=json.dumps(INITIAL_MANIFEST, indent=2).encode(),
        path_in_repo="manifest.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print("Uploaded manifest.json")


if __name__ == "__main__":
    main()
