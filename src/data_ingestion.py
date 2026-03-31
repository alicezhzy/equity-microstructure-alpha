from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_data(filename: str) -> pl.DataFrame:
    file_path = RAW_DIR / filename
    df = pl.read_parquet(file_path)
    return df


def inspect_schema(df: pl.DataFrame):
    print("\nColumns:")
    print(df.columns)

    print("\nSchema:")
    print(df.schema)

    print("\nFirst rows:")
    print(df.head())


def estimate_memory(df: pl.DataFrame):
    size_bytes = df.estimated_size()
    size_mb = size_bytes / (1024 ** 2)
    print(f"\nEstimated memory usage: {size_mb:.2f} MB")