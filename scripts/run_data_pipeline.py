import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_ingestion import load_raw_data
from src.data_cleaning import clean_trade_data, summarize_cleaning
from src.feature_engineering import (
    build_feature_table,
    add_lagged_returns,
    add_rolling_volatility,
    add_trade_intensity,
    add_volume_zscore,
)
from src.target_construction import add_returns, add_future_return, add_direction_label


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)

    raw_filename = config["raw_filename"]
    target_horizon = config["target_horizon"]
    filter_bad_jumps = config["filter_bad_jumps"]
    regular_trading_hours_only = config["regular_trading_hours_only"]
    output_filename = config["output_filename"]

    print("Loading raw data...")
    df = load_raw_data(raw_filename)
    print("Raw shape:", df.shape)

    print("Cleaning data...")
    clean_df = clean_trade_data(
        df,
        remove_dupes=True,
        filter_bad_prices=True,
        filter_bad_sizes=True,
        filter_bad_jumps=filter_bad_jumps,
        regular_trading_hours_only=regular_trading_hours_only,
    )
    print("Clean shape:", clean_df.shape)
    print(summarize_cleaning(df, clean_df))

    print("Building feature table...")
    feature_table = build_feature_table(clean_df)

    print("Adding targets...")
    feature_table = add_returns(feature_table)
    feature_table = add_future_return(feature_table, horizon=target_horizon)
    feature_table = add_direction_label(feature_table, horizon=target_horizon)

    print("Adding predictive features...")
    feature_table = add_lagged_returns(feature_table)
    feature_table = add_rolling_volatility(feature_table)
    feature_table = add_trade_intensity(feature_table)
    feature_table = add_volume_zscore(feature_table)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / output_filename

    print(f"Saving processed dataset to {output_path} ...")
    feature_table.write_parquet(output_path)

    print("Done.")
    print("Final shape:", feature_table.shape)
    print("Columns:", feature_table.columns)


if __name__ == "__main__":
    main()