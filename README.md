# fxquantbench/infra

Infrastructure repo for the FXQuantBench project — tick-data ingestion pipeline and strategy-runner Docker image.

## Contents

| Path | Purpose |
|------|---------|
| `docker/Dockerfile.strategy` | Base image for backtesting containers (`ghcr.io/fxquantbench/strategy-runner:latest`) |
| `ingest_historical.py` | Fetches GBPUSD tick data from Dukascopy and uploads to HuggingFace |
| `setup_hf_dataset.py` | One-time bootstrap: creates the HF dataset card and empty manifest |
| `validate.py` | Gap validation — exits non-zero if any 10-min gap exists in 07:00–22:00 UTC |
| `tests/` | Pytest suite (≥80% coverage enforced in CI) |
| `.github/workflows/` | CI test, Docker build, and daily ingest workflows |

## Quick Start

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- A HuggingFace write token with access to `FXQuantBench/fx-ticks`

### Install dependencies

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

### Bootstrap the HF dataset (one-time)

```bash
HF_TOKEN=<your-write-pat> uv run python setup_hf_dataset.py
```

### Run historical ingest

```bash
# Upload to HuggingFace (default: 2020-01-01 to end of last complete month)
HF_TOKEN=<your-write-pat> uv run python ingest_historical.py --start 2020-01-01

# Local-only (write Parquet files, skip HF upload)
uv run python ingest_historical.py --start 2024-01-01 --end 2024-01-31 --output-dir ./ticks

# Save pre-merge raw bid/ask snapshots alongside
HF_TOKEN=<your-write-pat> uv run python ingest_historical.py --start 2020-01-01 --raw-dir ./raw
```

### Run tests

```bash
uv run pytest tests/ --cov=ingest_historical --cov=setup_hf_dataset --cov=validate --cov-fail-under=80
```

## Data Schema

Parquet files stored at `GBPUSD/{YYYY}/{MM}/{DD}/ticks_{YYYY-MM-DD}.parquet` in `FXQuantBench/fx-ticks`.

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_utc` | int64 | Unix milliseconds, UTC |
| `bid` | float64 | Best bid price |
| `ask` | float64 | Best ask price |
| `bid_volume` | float32 | |
| `ask_volume` | float32 | |
| `is_interpolated` | bool | `True` if ask was forward-filled within 500 ms |

### Query with DuckDB

```sql
SELECT *
FROM read_parquet('hf://datasets/FXQuantBench/fx-ticks/GBPUSD/2023/01/01/ticks_2023-01-01.parquet')
LIMIT 10;
```

## CI / CD

| Workflow | Trigger | Action |
|----------|---------|--------|
| `test.yml` | Every push / PR | `pytest` with coverage gate (≥80%) |
| `build_docker.yml` | Push to `main` touching `docker/Dockerfile.strategy` | Build & push `strategy-runner` image to GHCR |
| `daily_ingest.yml` | Cron `0 8 * * 2-6` (Tue–Sat 08:00 UTC) | Ingest previous trading day → validate → upload to HF; opens GitHub issue on failure |

## Docker Image

```bash
docker pull ghcr.io/fxquantbench/strategy-runner:latest
```

Built from `python:3.13-slim` with: `pandas==2.2.3`, `numpy==1.26.4`, `pyarrow==18.1.0`, `duckdb==1.2.2`, `scipy==1.13.1`, `vectorbt==1.0.0`.
Runs as non-root user `runner` (UID 1000), working directory `/sandbox`.
