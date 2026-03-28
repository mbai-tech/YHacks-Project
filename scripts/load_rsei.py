"""Load, inspect, and clean RSEI (Risk-Screening Environmental Indicators) datasets.

Input files expected in DATA_RAW_DIR:
    facility.csv        — facility-level metadata (location, SIC, NAICS, etc.)
    releases.csv        — chemical release quantities per facility per year
    elements.csv        — chemical / element reference table
    water_microdata.csv — water-pathway micro-level score data (largest file)

Output:
    Cleaned Parquet files written to DATA_PROCESSED_DIR, one per dataset.

Usage:
    python scripts/load_rsei.py
    python scripts/load_rsei.py --raw-dir /path/to/raw --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
# dtype hints reduce memory and prevent silent mixed-type inference on large files.
# low_memory=False is the pandas default safety net; explicit dtypes are better.

DATASET_CONFIG: dict[str, dict] = {
    "facility": {
        "file": "facility.csv",
        "dtype": {
            "FacilityID":      "string",
            "TRIFacilityID":   "string",
            "FacilityName":    "string",
            "State":           "string",
            "County":          "string",
            "ZIPCode":         "string",
            "Latitude":        "float32",
            "Longitude":       "float32",
            "SICCode":         "string",
            "NAICSCode":       "string",
            "IndustryType":    "string",
        },
        # Columns that should become pandas nullable integers after load
        "int_cols": [],
        "date_cols": [],
        "chunksize": None,   # small enough to load in one shot
    },
    "releases": {
        "file": "releases.csv",
        "dtype": {
            "FacilityID":      "string",
            "CASNumber":       "string",
            "ChemicalID":      "string",
            "Year":            "Int16",
            "MediaCode":       "string",
            "ReleaseAmount":   "float64",
            "Units":           "string",
        },
        "int_cols": ["Year"],
        "date_cols": [],
        "chunksize": 200_000,
    },
    "elements": {
        "file": "elements.csv",
        "dtype": {
            "ChemicalID":      "string",
            "CASNumber":       "string",
            "ChemicalName":    "string",
            "ChemicalCategory":"string",
            "IsMetal":         "boolean",
        },
        "int_cols": [],
        "date_cols": [],
        "chunksize": None,
    },
    "water_microdata": {
        "file": "water_microdata.csv",
        "dtype": {
            "FacilityID":      "string",
            "ChemicalID":      "string",
            "Year":            "Int16",
            "CensusBlockID":   "string",
            "Population":      "Int32",
            "Concentration":   "float64",
            "RSEIScore":       "float64",
        },
        "int_cols": ["Year", "Population"],
        "date_cols": [],
        "chunksize": 500_000,   # largest file — stream in chunks
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_chunked(path: Path, config: dict) -> pd.DataFrame:
    """Read a large CSV in chunks and concatenate.

    Parameters
    ----------
    path:
        Path to the CSV file.
    config:
        Dataset config dict from DATASET_CONFIG.

    Returns
    -------
    pd.DataFrame
    """
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        dtype=config["dtype"],
        chunksize=config["chunksize"],
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",  # don't crash on stray non-UTF-8 bytes
        on_bad_lines="warn",
    )
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            rows_so_far = sum(len(c) for c in chunks)
            print(f"    ... {rows_so_far:,} rows loaded so far")

    return pd.concat(chunks, ignore_index=True)


def _read_single(path: Path, config: dict) -> pd.DataFrame:
    """Read a small CSV in one shot."""
    return pd.read_csv(
        path,
        dtype=config["dtype"],
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
    )


def _clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply post-load cleaning common to all datasets.

    Steps
    -----
    1. Strip leading/trailing whitespace from all string columns.
    2. Normalise column names to snake_case.
    3. Drop fully-duplicate rows.
    4. Cast declared integer columns to pandas nullable Int type.
    5. Parse declared date columns.
    """
    # 1. Strip whitespace in string/object columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2. snake_case column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    # 3. Drop fully-duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped:,} duplicate rows")

    # 4. Nullable integer casts (column names already snake_cased above)
    for col in config["int_cols"]:
        snake_col = col.lower()
        if snake_col in df.columns:
            df[snake_col] = pd.to_numeric(df[snake_col], errors="coerce").astype("Int64")

    # 5. Date parsing
    for col in config["date_cols"]:
        snake_col = col.lower()
        if snake_col in df.columns:
            df[snake_col] = pd.to_datetime(df[snake_col], errors="coerce")

    return df


def _print_info(name: str, df: pd.DataFrame) -> None:
    """Print column names, dtypes, shape, and null counts."""
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}  —  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'=' * 60}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
    print()

    col_width = max(len(c) for c in df.columns) + 2
    print(f"  {'Column':<{col_width}} {'Dtype':<14} {'Non-null':>10}  {'Null%':>6}")
    print(f"  {'-' * (col_width + 35)}")
    for col in df.columns:
        n_null = df[col].isna().sum()
        null_pct = 100 * n_null / len(df) if len(df) else 0
        print(
            f"  {col:<{col_width}} {str(df[col].dtype):<14} "
            f"{len(df) - n_null:>10,}  {null_pct:>5.1f}%"
        )

    print()
    # Numeric summaries for key float/int columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        print("  Numeric summary:")
        print(
            df[num_cols]
            .describe()
            .round(4)
            .to_string(max_cols=8)
            .replace("\n", "\n  ")
        )
    print()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all(raw_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Load, clean, inspect, and save all RSEI datasets.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw CSV files.
    out_dir:
        Directory where Parquet outputs are written.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of dataset name → cleaned DataFrame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for name, config in DATASET_CONFIG.items():
        path = raw_dir / config["file"]

        if not path.exists():
            print(f"\n[SKIP] {config['file']} not found in {raw_dir}")
            continue

        print(f"\n[LOAD] {name}  ← {path.name}")
        t0 = time.perf_counter()

        if config["chunksize"]:
            df = _read_chunked(path, config)
        else:
            df = _read_single(path, config)

        elapsed_read = time.perf_counter() - t0
        print(f"    Read  {len(df):,} rows in {elapsed_read:.1f}s")

        df = _clean(df, config)

        _print_info(name, df)

        # Save to Parquet
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  Saved → {out_path.relative_to(out_dir.parent.parent)}  ({size_mb:.1f} MB)")

        results[name] = df

    print(f"\n{'=' * 60}")
    print(f"  Done.  {len(results)}/{len(DATASET_CONFIG)} datasets loaded.")
    print(f"  Parquet files in: {out_dir}")
    print(f"{'=' * 60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and clean RSEI CSV datasets.")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help=f"Directory containing raw CSVs (default: {DATA_RAW_DIR})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help=f"Output directory for Parquet files (default: {DATA_PROCESSED_DIR})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load_all(raw_dir=args.raw_dir, out_dir=args.out_dir)
