"""Load, inspect, and clean RSEI v23.12 datasets into Parquet.

Input files expected in DATA_RAW_DIR:
    aggmicro2022_2022.csv          — grid-cell RSEI scores (~10M rows)
    facility_data_rsei_v2312.csv   — facility metadata (~64k rows)
    chemical_data_rsei_v2312.csv   — chemical reference table (~823 rows)
    media_data_rsei_v2312.csv      — release media lookup (~47 rows)

Output:
    Cleaned Parquet files written to DATA_PROCESSED_DIR.

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
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIG: dict[str, dict] = {

    "aggmicro": {
        "file": "aggmicro2022_2022.csv",
        "encoding": "utf-8",
        "dtype": {
            "GridCode":    "int8",      # always 14 in this file
            "X":           "int16",
            "Y":           "int16",
            "NumFacs":     "int16",
            "NumReleases": "int32",
            "NumChems":    "int16",
            "ToxConc":     "float64",
            "Score":       "float64",
            "Pop":         "float32",
            "CTConc":      "float64",
            "NCTConc":     "float64",
        },
        "chunksize": 500_000,
    },

    "facility": {
        "file": "facility_data_rsei_v2312.csv",
        "encoding": "utf-8",
        "dtype": {
            "FacilityID":               "string",
            "FacilityNumber":           "Int32",
            "FRSID":                    "Int64",
            "Latitude":                 "float32",
            "Longitude":                "float32",
            "GridCode":                 "Int8",
            "X":                        "Int16",
            "Y":                        "Int16",
            "LatLongSource":            "string",
            "LLYear":                   "Int16",
            "LLNotes":                  "string",
            "RadialDistance":           "float32",
            "FacilityName":             "string",
            "Street":                   "string",
            "City":                     "string",
            "County":                   "string",
            "State":                    "string",
            "ZIPCode":                  "string",    # keep as string — leading zeros
            "ZIP9":                     "string",
            "FIPS":                     "string",
            "STFIPS":                   "string",
            "DUNS":                     "string",
            "ParentDUNS":               "string",
            "ParentName":               "string",
            "StandardizedParentCompany":"string",
            "Region":                   "Int8",
            "FederalFacilityFlag":      "string",
            "FederalAgencyName":        "string",
            "PublicContactName":        "string",
            "PublicContactPhone":       "string",
            "PublicContactPhoneExtension": "string",
            "PublicContactEmail":       "string",
            "StackHeight":              "float32",
            "StackVelocity":            "float32",
            "StackDiameter":            "float32",
            "StackTemperature":         "float32",
            "StackHeightSource":        "string",
            "StackVelocitySource":      "string",
            "StackDiameterSource":      "string",
            "StackTemperatureSource":   "string",
            "StackHeightNEIYear":       "Int16",
            "StackVelocityNEIYear":     "Int16",
            "StackDiameterNEIYear":     "Int16",
            "StackTemperatureNEIYear":  "Int16",
            "ChromHexPercent":          "float32",
            "ChromSource":              "string",
            "ModeledNAICS":             "string",
            "NAICSCode3Digit":          "string",
            "NAICSCode4Digit":          "string",
            "NAICSCode5Digit":          "string",
            "NAICS1":                   "string",
            "NAICS2":                   "string",
            "NAICS3":                   "string",
            "NAICS4":                   "string",
            "NAICS5":                   "string",
            "NAICS6":                   "string",
            "HEM3ID":                   "Int32",
            "DistanceToHEM3":           "float32",
            "LLConfirmed":              "boolean",
            "WaterReleases":            "boolean",
            "ModeledReleases":          "boolean",
        },
        "chunksize": None,
    },

    "chemical": {
        "file": "chemical_data_rsei_v2312.csv",
        "encoding": "utf-8-sig",          # file has BOM
        "dtype": {
            "CASNumber":            "string",
            "CASStandard":          "string",
            "ChemicalNumber":       "Int16",
            "TRIChemID":            "string",
            "Chemical":             "string",
            "FirstReportingYear":   "Int16",
            "ToxicitySource":       "string",
            "RfCInhale":            "float64",
            "RfCUF":                "float64",
            "RfCMF":                "float64",
            "RfCConf":              "string",
            "RfCSource":            "string",
            "RfCToxWeight":         "float64",
            "RFDOral":              "float64",
            "RfDUF":                "float64",
            "RfDMF":                "float64",
            "RfDConf":              "string",
            "RfDSource":            "string",
            "RfDToxWeight":         "float64",
            "UnitRiskInhale":       "float64",
            "QSTAROral":            "float64",
            "WOE":                  "string",
            "IURToxWeight":         "float64",
            "OSFToxWeight":         "float64",
            "ITW":                  "float64",
            "OTW":                  "float64",
            "ToxicityClassOral":    "string",
            "ToxicityClassInhale":  "string",
            "ToxicityCategory":     "string",
            "AirDecay":             "float64",
            "Koc":                  "float64",
            "H2ODecay":             "float64",
            "LOGKow":               "float64",
            "Kd":                   "float64",
            "WaterSolubility":      "float64",
            "POTWPartitionRemoval": "float64",
            "POTWPartitionSludge":  "float64",
            "POTWPartitionVolat":   "float64",
            "POTWPartitionBiod":    "float64",
            "IncineratorDRE":       "float64",
            "BCF":                  "float64",
            "Henrys":               "float64",
            "MCL":                  "float64",
            "MolecularWeight":      "float64",
            "HAPFlag":              "boolean",
            "RMPFlag":              "boolean",
            "PriorityPollutantFlag":"boolean",
            "SDWAFlag":             "boolean",
            "CERCLAFlag":           "boolean",
            "OSHACarcinogens":      "boolean",
            "ExpansionFlag":        "boolean",
            "PBTFlag":              "boolean",
            "TSCAFlag":             "boolean",
            "PFASFlag":             "boolean",
            "Metal":                "Int8",
            "HasTox":               "Int8",
            "MaxTW":                "float64",
            "MaxNC":                "float64",
            "MaxC":                 "float64",
        },
        "chunksize": None,
    },

    "media": {
        "file": "media_data_rsei_v2312.csv",
        "encoding": "utf-8",
        "dtype": {
            "Media":                    "Int8",
            "MediaText":                "string",
            "TRIEnvironmentalMedium":   "string",
            "Itw":                      "Int8",
            "Otw":                      "Int8",
            "Mtw":                      "Int8",
            "MediaCode":                "Int8",
            "TRICode":                  "string",
            "TRICategory":              "string",
            "LongDescription":          "string",
            "AggDescription":           "string",
        },
        "chunksize": None,
    },

    # ----- CWNS wastewater plant tables -----

    "cwns_facilities": {
        "file": "FACILITIES.csv",
        "encoding": "utf-8",
        "dtype": {
            "CWNS_ID":              "string",
            "FACILITY_ID":          "string",
            "STATE_CODE":           "string",
            "INFRASTRUCTURE_TYPE":  "string",
            "FACILITY_NAME":        "string",
            "DESCRIPTION":          "string",
            "OWNER_TYPE":           "string",
            "SUPERFUND_FLAG":       "string",
            "SEMS_ID":              "string",
            "SCF_ELIGIBLE":         "string",
            "REVIEW_STATUS":        "string",
            "REVIEW_TYPE":          "string",
            "DATE_LAST_MODIFIED":   "string",
            "NO_NEEDS":             "string",
        },
        "chunksize": None,
    },

    "cwns_locations": {
        "file": "PHYSICAL_LOCATION.csv",
        "encoding": "utf-8",
        "dtype": {
            "CWNS_ID":        "string",
            "FACILITY_ID":    "string",
            "LOCATION_TYPE":  "string",
            "LATITUDE":       "float64",
            "LONGITUDE":      "float64",
            "DATUM":          "string",
            "ADDRESS":        "string",
            "ADDRESS_2":      "string",
            "CITY":           "string",
            "STATE_CODE":     "string",
            "ZIP_CODE":       "string",
            "COUNTY_FIPS":    "string",
            "COUNTY_NAME":    "string",
        },
        "chunksize": None,
    },

    "cwns_population": {
        "file": "POPULATION_WASTEWATER.csv",
        "encoding": "utf-8",
        "dtype": {
            "CWNS_ID":                              "string",
            "FACILITY_ID":                          "string",
            "STATE_CODE":                           "string",
            "INFRASTRUCTURE_TYPE":                  "string",
            "POPULATION_TYPE":                      "string",
            "RESIDENTIAL_POP_2022":                 "float64",
            "RESIDENTIAL_POP_2042":                 "float64",
            "NONRESIDENTIAL_POP_2022":              "float64",
            "NONRESIDENTIAL_POP_2042":              "float64",
            "TOTAL_RES_POPULATION_2022":            "float64",
            "TOTAL_RES_POPULATION_2042":            "float64",
            "TOTAL_NONRES_POPULATION_2022":         "float64",
            "TOTAL_NONRES_POPULATION_2042":         "float64",
            "TREATED_DISCHARGE_PERCENTAGE_2022":    "float64",
            "TREATED_DISCHARGE_PERCENTAGE_2042":    "float64",
            "NET_TREATED_POPULATION_2022":          "float64",
            "NET_TREATED_POPULATION_2042":          "float64",
            "NO_DISCHARGE_PERCENTAGE_2022":         "float64",
            "NO_DISCHARGE_PERCENTAGE_2042":         "float64",
            "NO_DISCHARGE_POPULATION_2022":         "float64",
            "NO_DISCHARGE_POPULATION_2042":         "float64",
            "PART_OF_SEWERSHED":                    "string",
            "END_FACILITY":                         "string",
        },
        "chunksize": None,
    },

    "cwns_discharges": {
        "file": "DISCHARGES.csv",
        "encoding": "utf-8",
        "dtype": {
            "CWNS_ID":                         "string",
            "FACILITY_ID":                     "string",
            "STATE_CODE":                      "string",
            "DISCHARGE_TYPE":                  "string",
            "PRESENT_DISCHARGE_PERCENTAGE":    "float64",
            "PROJECTED_DISCHARGE_PERCENTAGE":  "float64",
            "DISCHARGES_TO_CWNSID":            "string",
        },
        "chunksize": None,
    },

    # ----- SDWA (Safe Drinking Water Act) tables -----

    "sdwa_water_systems": {
        "file": "SDWA_PUB_WATER_SYSTEMS.csv",
        "encoding": "utf-8",
        "dtype": "string",       # read everything as string, coerce later
        "coerce_numeric": ["POPULATION_SERVED_COUNT", "SERVICE_CONNECTIONS_COUNT"],
        "chunksize": None,
    },

    "sdwa_violations": {
        "file": "SDWA_PN_VIOLATION_ASSOC.csv",
        "encoding": "utf-8",
        "dtype": "string",
        "chunksize": None,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path, config: dict) -> pd.DataFrame:
    """Read a CSV — chunked if chunksize is set, single-pass otherwise."""
    common = dict(
        dtype=config["dtype"],
        low_memory=False,
        encoding=config.get("encoding", "utf-8"),
        encoding_errors="replace",
        on_bad_lines="warn",
    )

    if config["chunksize"]:
        chunks: list[pd.DataFrame] = []
        reader = pd.read_csv(path, chunksize=config["chunksize"], **common)
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                print(f"    ... {sum(len(c) for c in chunks):,} rows loaded")
        return pd.concat(chunks, ignore_index=True)

    return pd.read_csv(path, **common)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-load cleaning: strip whitespace, drop full duplicates."""
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped:,} duplicate rows")
    return df


def _print_info(name: str, df: pd.DataFrame) -> None:
    """Print shape, memory, and a per-column summary."""
    print(f"\n{'=' * 62}")
    print(f"  {name.upper()}  —  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
    print(f"{'=' * 62}")

    col_w = max(len(c) for c in df.columns) + 2
    print(f"  {'Column':<{col_w}} {'Dtype':<12} {'Non-null':>10}  {'Null%':>6}")
    print(f"  {'-' * (col_w + 33)}")
    for col in df.columns:
        n_null = df[col].isna().sum()
        pct = 100 * n_null / len(df) if len(df) else 0
        print(f"  {col:<{col_w}} {str(df[col].dtype):<12} {len(df)-n_null:>10,}  {pct:>5.1f}%")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print()
        print("  Numeric summary:")
        summary = df[num_cols].describe().round(4)
        print("  " + summary.to_string().replace("\n", "\n  "))
    print()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all(raw_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Load, clean, inspect, and save all four RSEI datasets.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw CSV files.
    out_dir : Path
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

        print(f"\n[LOAD] {name}  ←  {path.name}")
        t0 = time.perf_counter()
        df = _read(path, config)
        print(f"    Read {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

        df = _clean(df)
        _print_info(name, df)

        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  Saved → {out_path.name}  ({size_mb:.1f} MB)")

        results[name] = df

    print(f"\n{'=' * 62}")
    print(f"  Done. {len(results)}/{len(DATASET_CONFIG)} datasets loaded.")
    print(f"  Parquet files in: {out_dir}")
    print(f"{'=' * 62}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and clean RSEI v23.12 CSV datasets.")
    p.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DATA_PROCESSED_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load_all(raw_dir=args.raw_dir, out_dir=args.out_dir)
